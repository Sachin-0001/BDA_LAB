"""Train the default-risk classifier from the data stored on HDFS.

Time-based split (train on older loans, test on the newest 20%) to mimic real deployment.
Outputs models/model.joblib and models/metrics.json.

    python -m lrisk.train                      # read HDFS
    python -m lrisk.train --local data/x.csv   # skip HDFS (debug)
"""
import argparse
import io
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve

from . import config
from .features import (CATEGORICAL, DATE_COL, FEATURES, ID_COL, LABEL_COL, RAW_COLUMNS, build_features,
                       category_vocab, make_target)
from .hdfs_client import HDFS
from .model import ModelBundle, save


def load_from_hdfs() -> pd.DataFrame:
    hdfs = HDFS()
    files = [f for f in hdfs.walk_files(config.HDFS_RAW_DIR) if f.endswith(".jsonl")]
    if not files:
        raise SystemExit(f"no .jsonl files under {config.HDFS_RAW_DIR} - run the producer and sink first")
    frames = [pd.read_json(io.BytesIO(hdfs.read(f)), lines=True, dtype=False) for f in files]
    print(f"read {len(files)} files from HDFS")
    return pd.concat(frames, ignore_index=True)


def metrics_at(y, p, thr) -> dict:
    flag = p > thr
    tp = int((flag & (y == 1)).sum())
    return {
        "threshold": thr,
        "flag_rate": float(flag.mean()),
        "precision": float(tp / flag.sum()) if flag.sum() else 0.0,
        "recall": float(tp / (y == 1).sum()),
    }


def capture_at_review_rate(y, score, rate) -> float:
    """Share of all defaults caught when the top `rate` fraction of applicants is reviewed."""
    k = max(1, int(round(rate * len(y))))
    idx = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(y)[idx].sum() / np.asarray(y).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="read this CSV instead of HDFS")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=config.REVIEW_THRESHOLD)
    a = ap.parse_args()

    raw = pd.read_csv(a.local, usecols=lambda c: c in RAW_COLUMNS, low_memory=False) if a.local else load_from_hdfs()
    n_in = len(raw)
    raw = raw.drop_duplicates(subset=ID_COL, keep="last")
    y = make_target(raw[LABEL_COL])
    keep = y.notna()
    raw, y = raw[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)
    print(f"{n_in:,} rows ingested -> {len(raw):,} unique loans with a known outcome; default rate {y.mean():.1%}")

    order = pd.to_datetime(raw[DATE_COL], format="%b-%Y", errors="coerce").fillna(pd.Timestamp("1970-01-01")).argsort(kind="stable")
    raw, y = raw.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    X = build_features(raw)
    cats = category_vocab(X)
    cut = int(len(X) * (1 - a.test_frac))
    Xtr, Xte, ytr, yte = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    # Validation slice for early stopping = last 15% of the training period (still earlier than test).
    v = int(len(Xtr) * 0.85)
    dtr = lgb.Dataset(Xtr.iloc[:v], ytr.iloc[:v], categorical_feature=CATEGORICAL)
    dva = lgb.Dataset(Xtr.iloc[v:], ytr.iloc[v:], reference=dtr)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  cat_smooth=20, verbose=-1, seed=42)
    booster = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    p = booster.predict(Xte)
    _fpr, _tpr, _ = roc_curve(yte, p)

    # The baseline the problem statement describes: a single fixed credit-score cutoff.
    fico_risk = -Xte["fico"].fillna(Xte["fico"].median()).to_numpy()
    rate = float((p > a.threshold).mean())
    metrics = {
        "rows_total": int(n_in), "rows_trained": int(len(Xtr)), "rows_test": int(len(Xte)),
        "default_rate_test": float(yte.mean()),
        "best_iteration": int(booster.best_iteration),
        "auc": float(roc_auc_score(yte, p)),
        "pr_auc": float(average_precision_score(yte, p)),
        "ks": float(np.max(_tpr - _fpr)),
        "log_loss": float(log_loss(yte, p)), "brier": float(brier_score_loss(yte, p)),
        "at_threshold": metrics_at(yte.to_numpy(), p, a.threshold),
        "baseline_fico_auc": float(roc_auc_score(yte, fico_risk)),
        "capture_at_model_review_rate": {
            "review_rate": rate,
            "model": capture_at_review_rate(yte, p, rate) if rate else 0.0,
            "fico_cutoff": capture_at_review_rate(yte, fico_risk, rate) if rate else 0.0,
        },
        "capture_at_20pct_review": {
            "model": capture_at_review_rate(yte, p, 0.2), "fico_cutoff": capture_at_review_rate(yte, fico_risk, 0.2),
        },
        "feature_importance_gain": dict(sorted(
            zip(FEATURES, map(float, booster.feature_importance("gain"))), key=lambda kv: -kv[1])),
    }
    save(ModelBundle(booster, cats, a.threshold, meta={"metrics_summary": {"auc": metrics["auc"]}}))
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    t = metrics["at_threshold"]
    print(f"\nTest AUC {metrics['auc']:.3f} | PR-AUC {metrics['pr_auc']:.3f} | KS {metrics['ks']:.3f} | FICO-only AUC {metrics['baseline_fico_auc']:.3f}")
    print(f"At threshold {a.threshold}: flags {t['flag_rate']:.1%} of applications, precision {t['precision']:.1%}, recall {t['recall']:.1%}")
    c = metrics["capture_at_20pct_review"]
    print(f"Reviewing the riskiest 20%: model catches {c['model']:.1%} of defaults vs {c['fico_cutoff']:.1%} for a fixed FICO cutoff")
    print(f"saved {config.MODEL_PATH} and {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
