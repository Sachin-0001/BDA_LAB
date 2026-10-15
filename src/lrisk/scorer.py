"""Real-time scorer: Kafka -> default probability -> SQLite (dashboard) + review topic.

Scores above REVIEW_THRESHOLD (default 0.5) are flagged for manual underwriting and also
published to the `underwriting-review` topic for downstream systems.

    python -m lrisk.scorer                   # only new messages
    python -m lrisk.scorer --from-beginning  # replay the whole topic
"""
import argparse
import json
import time

import pandas as pd
from confluent_kafka import Consumer, Producer

from . import config, store
from .features import build_features
from .kafka_utils import ensure_topics
from .model import load


def to_rows(bundle, records: list[dict]) -> tuple[list[dict], list[dict]]:
    raw = pd.DataFrame(records)
    probs, reasons = bundle.score(raw)
    feats = build_features(raw, bundle.categories, now=pd.Timestamp.now().normalize())
    rows, review_events = [], []
    for i, rec in enumerate(records):
        flagged = bool(probs[i] > bundle.threshold)
        row = {
            "application_id": str(rec["id"]),
            "loan_amnt": _f(rec.get("loan_amnt")), "annual_inc": _f(rec.get("annual_inc")),
            "dti": _f(rec.get("dti")), "fico": _f(feats["fico"].iloc[i]),
            "term": (rec.get("term") or "").strip(), "purpose": rec.get("purpose"),
            "score": float(probs[i]), "flagged": int(flagged), "reasons": reasons[i],
            "review_status": "pending" if flagged else "not_required",
        }
        rows.append(row)
        if flagged:
            review_events.append({k: row[k] for k in ("application_id", "score", "reasons")})
    return rows, review_events


def _f(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-beginning", action="store_true")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--batch-seconds", type=float, default=1.0)
    ap.add_argument("--idle-exit", type=float, default=0)
    ap.add_argument("--group", default="scorer")
    a = ap.parse_args()

    bundle = load()
    print(f"model loaded; review threshold {bundle.threshold}")
    ensure_topics()
    c = Consumer({"bootstrap.servers": config.KAFKA_BOOTSTRAP, "group.id": a.group,
                  "auto.offset.reset": "earliest" if a.from_beginning else "latest",
                  "enable.auto.commit": False})
    c.subscribe([config.TOPIC_APPLICATIONS])
    prod = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP})

    buf: list[dict] = []
    n_scored = n_flagged = 0
    last_flush = last_msg = time.time()

    def flush() -> None:
        nonlocal buf, n_scored, n_flagged, last_flush
        if buf:
            rows, events = to_rows(bundle, buf)
            store.upsert_scores(rows)
            for e in events:
                prod.produce(config.TOPIC_REVIEW, key=e["application_id"], value=json.dumps(e))
            prod.flush()
            c.commit(asynchronous=False)
            n_scored += len(rows)
            n_flagged += len(events)
            print(f"scored {len(rows)} (total {n_scored}, flagged {n_flagged})")
        buf = []
        last_flush = time.time()

    try:
        while True:
            m = c.poll(0.5)
            now = time.time()
            if m is None or m.error():
                if buf and now - last_flush >= a.batch_seconds:
                    flush()
                if a.idle_exit and now - last_msg >= a.idle_exit:
                    break
                continue
            last_msg = now
            buf.append(json.loads(m.value()))
            if len(buf) >= a.batch_size or now - last_flush >= a.batch_seconds:
                flush()
    except KeyboardInterrupt:
        pass
    finally:
        flush()
        c.close()


if __name__ == "__main__":
    main()
