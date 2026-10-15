# Loan Default Risk Prediction

Replaces a fixed credit-score cutoff with a model that scores each loan application on the
*combined* effect of income, debt ratio, credit history and loan terms. Applications with a
default probability **> 0.5** are flagged for manual underwriting in a dashboard.

```
 Lending Club CSV
       │  producer.py
       ▼
 Kafka  topic: loan-applications ──┬──► hdfs_sink.py ──► HDFS /lending_club/raw/ingest_date=…/*.jsonl
                                   │                              │
                                   │                              ▼  train.py (LightGBM, time-based split)
                                   │                        models/model.joblib + metrics.json
                                   │                              │
                                   └──► scorer.py ◄───────────────┘
                                          │  default probability + top-3 risk drivers
                                          ├──► SQLite ──► dashboard (Flask, :5055)  approve / decline / need info
                                          └──► Kafka topic: underwriting-review  (flagged only)
```

## Quick start (synthetic data, ~2 min)

```bash
make demo        # starts Kafka+HDFS, generates sample, loads, trains, streams live applications
make dashboard   # open http://localhost:5055
make test
```

## Using the real Lending Club dataset

The Kaggle dataset needs your Kaggle login, so it is **not** bundled.

1. Kaggle → Settings → *Create New Token* → save as `~/.kaggle/kaggle.json`
2. `scripts/download_dataset.sh` (downloads `wordsforthewise/lending-club`, ~375 MB), or download
   it manually and put `accepted_2007_to_2018Q4.csv.gz` in `data/`.
3. ```bash
   make up
   make load CSV=data/lending_club/accepted_2007_to_2018Q4.csv.gz   # loans issued before 2017 → Kafka → HDFS
   make train
   make live CSV=data/lending_club/accepted_2007_to_2018Q4.csv.gz   # 2017+ loans, label stripped, scored live
   make dashboard
   ```
The older `loan.csv` (2007–2015) layout should also load (columns it lacks become null, e.g. FICO — check the producer's "columns not in file" note); pass `SPLIT=2015-01` or similar.
The full file is ~2.2M rows; loading takes several minutes. Use `--limit` on the producer to sample.

## Design decisions

- **Target**: `Charged Off` / `Default` = 1, `Fully Paid` = 0. `Current`, `Late` etc. are dropped (outcome unknown).
- **No leakage**: only application-time fields are used. Post-origination columns (`total_pymnt`,
  `recoveries`, …) are excluded. Lender-assigned `grade` / `sub_grade` / `int_rate` are also excluded —
  they are the lender's own prior risk opinion; add them in `features.py` if you want them.
- **Combined effects**: engineered `payment_to_income`, `loan_to_income`, `revol_bal_to_income`,
  `credit_history_years`; gradient boosting captures the interactions (e.g. DTI hurting more at low FICO).
- **Time-based split**: train on older loans, test on the newest 20% — what a deployed model actually faces.
- **Reason codes**: each flagged application shows its top 3 risk-increasing drivers (LightGBM SHAP contributions).
- **Baseline**: `metrics.json` compares against a FICO-only cutoff, the status quo in the problem statement.
- **Delivery semantics**: sink and scorer commit Kafka offsets after the write (at-least-once);
  HDFS file names derive from offsets and SQLite upserts by application id, so replays don't duplicate.
  The trainer also de-duplicates by id.

## The 0.5 threshold

`REVIEW_THRESHOLD` (env var, default `0.5`) is the flag cutoff, as specified. Defaults are a minority
class (~20% of resolved Lending Club loans), so few applications ever score above 0.5: the review queue
is small and high-precision, but **recall is low**. `metrics.json` and the dashboard show flag rate,
precision and recall at the cutoff. If underwriting capacity allows, a lower threshold (e.g. 0.3) catches
more defaults; choose it from the volume you can review. Changing it only needs a scorer restart.

## Layout

| Path | Purpose |
|---|---|
| `docker-compose.yml`, `hadoop.env` | Kafka (KRaft) + HDFS NameNode/DataNode. UIs: HDFS http://localhost:9870 |
| `src/lrisk/producer.py` | CSV → Kafka (`--live` strips the label, `--rate` throttles) |
| `src/lrisk/hdfs_sink.py` | Kafka → HDFS micro-batches via WebHDFS |
| `src/lrisk/train.py` | HDFS → features → LightGBM → `models/` |
| `src/lrisk/scorer.py` | Kafka → score → SQLite + `underwriting-review` topic |
| `src/lrisk/dashboard/` | Review queue UI and JSON API (`/api/stats`, `/api/flagged`) |
| `src/lrisk/generate_sample.py` | **Synthetic** Lending Club-format data for demos/tests only |

## Notes

- Dashboard binds to 127.0.0.1 and has no authentication or reviewer identity; add both before real use.
- Single-node Kafka/HDFS with replication 1 — a dev setup, not production.
- Metrics produced from the synthetic sample reflect the generator, not real credit behaviour.
  Re-run `make train` on the real data for meaningful numbers.
