"""Offline tests (no Kafka / HDFS needed)."""
import numpy as np
import pandas as pd
import pytest

from lrisk import store
from lrisk.features import FEATURES, build_features, category_vocab, make_target, parse_emp_length
from lrisk.generate_sample import generate


def test_make_target_maps_outcomes_and_ignores_unresolved():
    s = pd.Series(["Fully Paid", "Charged Off", "Default", "Current", "Late (31-120 days)",
                   "Does not meet the credit policy. Status:Charged Off"])
    y = make_target(s)
    assert y.tolist()[:3] == [0.0, 1.0, 1.0]
    assert y.iloc[3:5].isna().all()
    assert y.iloc[5] == 1.0


def test_emp_length_parsing():
    out = parse_emp_length(pd.Series(["10+ years", "< 1 year", "3 years", None]))
    assert out.iloc[:3].tolist() == [10.0, 0.0, 3.0] and np.isnan(out.iloc[3])


def test_build_features_formats_and_engineering():
    raw = pd.DataFrame([{
        "id": "1", "term": " 60 months", "emp_length": "5 years", "issue_d": "Dec-2015",
        "earliest_cr_line": "Dec-2005", "annual_inc": 60000, "installment": 500.0, "loan_amnt": 24000,
        "revol_bal": 6000, "revol_util": "55.5%", "fico_range_low": 700, "fico_range_high": 704,
        "home_ownership": "RENT", "purpose": "credit_card",
    }])
    X = build_features(raw)
    assert list(X.columns) == FEATURES
    r = X.iloc[0]
    assert r.term_months == 60 and r.fico == 702 and r.revol_util == 55.5
    assert r.payment_to_income == pytest.approx(0.1) and r.loan_to_income == pytest.approx(0.4)
    assert r.credit_history_years == pytest.approx(10, abs=0.05)


def test_unseen_category_becomes_missing_not_error():
    train = build_features(generate(500))
    vocab = category_vocab(train)
    odd = generate(5)
    odd["purpose"] = "moon_base"
    assert build_features(odd, vocab)["purpose"].isna().all()


def test_zero_income_does_not_explode():
    raw = generate(3)
    raw["annual_inc"] = 0
    X = build_features(raw)
    assert X["payment_to_income"].isna().all()


def test_scoring_flags_only_above_threshold_and_store_roundtrip(tmp_path):
    import lightgbm as lgb
    from lrisk.model import ModelBundle
    from lrisk.scorer import to_rows
    from lrisk.features import CATEGORICAL

    raw = generate(4000)
    raw = raw[raw.loan_status.isin(["Fully Paid", "Charged Off"])]
    X = build_features(raw)
    y = make_target(raw.loan_status).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 15},
                        lgb.Dataset(X, y, categorical_feature=CATEGORICAL), num_boost_round=40)
    bundle = ModelBundle(booster, category_vocab(X), threshold=0.5)

    recs = generate(50, seed=99).drop(columns=["loan_status"]).to_dict("records")
    rows, events = to_rows(bundle, recs)
    assert len(rows) == 50
    for r in rows:
        assert 0 <= r["score"] <= 1
        assert r["flagged"] == int(r["score"] > 0.5)
        assert r["review_status"] == ("pending" if r["flagged"] else "not_required")
    assert len(events) == sum(r["flagged"] for r in rows)

    db = tmp_path / "t.db"
    store.upsert_scores(rows, db)
    store.upsert_scores(rows, db)  # idempotent on replay
    assert store.stats(db)["total"] == 50
    flagged = store.list_flagged("pending", path=db)
    if flagged:
        assert store.decide(flagged[0]["application_id"], "approved", "ok", db)
        assert store.stats(db)["by_status"].get("approved") == 1
    unflagged = [r for r in rows if not r["flagged"]]
    assert unflagged and not store.decide(unflagged[0]["application_id"], "approved", path=db)
    with pytest.raises(ValueError):
        store.decide("x", "maybe", path=db)


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    from lrisk import config
    from lrisk.dashboard.app import app
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "d.db")
    rows = [{"application_id": f"A{i}", "loan_amnt": 10000.0, "annual_inc": 50000.0, "dti": 20.0, "fico": 650.0,
             "term": " 36 months", "purpose": "credit_card", "score": s, "flagged": int(s > 0.5),
             "reasons": [{"feature": "FICO score", "value": "650", "impact": 0.8},
                         {"feature": "Debt-to-income ratio", "value": "20", "impact": 0.2}] if s > 0.5 else [],
             "review_status": "pending" if s > 0.5 else "not_required"}
            for i, s in enumerate([0.1, 0.3, 0.55, 0.65, 0.9])]
    store.upsert_scores(rows)
    return app.test_client()


def test_dashboard_renders_every_tab(dashboard):
    for status in ["pending", "approved", "declined", "more_info", "all", "bogus"]:
        r = dashboard.get(f"/?status={status}")
        assert r.status_code == 200, status
    html = dashboard.get("/").get_data(as_text=True)
    assert "A4" in html and "A0" not in html and "sev-high" in html and "Debt-to-income ratio" in html


def test_dashboard_renders_without_metrics_or_data(tmp_path, monkeypatch):
    from lrisk import config
    from lrisk.dashboard.app import app
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "empty.db")
    monkeypatch.setattr(config, "METRICS_PATH", tmp_path / "missing.json")
    html = app.test_client().get("/").get_data(as_text=True)
    assert "Queue is clear" in html and "No model metrics yet" in html


def test_review_json_and_form_fallback(dashboard):
    r = dashboard.post("/review/A4", data={"decision": "declined", "note": " thin file "},
                       headers={"Accept": "application/json"})
    assert r.status_code == 200 and r.json["ok"] and r.json["stats"]["by_status"]["declined"] == 1
    assert store.list_flagged("declined")[0]["reviewer_note"] == "thin file"
    assert dashboard.post("/review/A0", data={"decision": "approved"},
                          headers={"Accept": "application/json"}).status_code == 404  # not flagged
    assert dashboard.post("/review/A4", data={"decision": "maybe"}).status_code == 400
    r = dashboard.post("/review/A3?status=pending", data={"decision": "approved"})
    assert r.status_code == 302 and "status=pending" in r.headers["Location"]
    assert "“thin file”" in dashboard.get("/?status=declined").get_data(as_text=True)
