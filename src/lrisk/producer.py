"""Stream loan applications from a Lending Club CSV (plain or .gz) into Kafka.

    # history (with outcome label) -> used for training
    python -m lrisk.producer data/accepted_2007_to_2018Q4.csv.gz --issued-before 2017-01
    # live applications (label stripped) -> scored in real time
    python -m lrisk.producer data/accepted_2007_to_2018Q4.csv.gz --issued-after 2017-01 --live --rate 20
"""
import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer

from . import config
from .features import DATE_COL, ID_COL, LABEL_COL, RAW_COLUMNS
from .kafka_utils import ensure_topics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--live", action="store_true", help="strip loan_status (applications awaiting a decision)")
    ap.add_argument("--issued-before", help="YYYY-MM, exclusive")
    ap.add_argument("--issued-after", help="YYYY-MM, inclusive")
    ap.add_argument("--limit", type=int, default=0, help="max messages (0 = all)")
    ap.add_argument("--rate", type=float, default=0, help="messages/second (0 = as fast as possible)")
    ap.add_argument("--chunksize", type=int, default=50_000)
    a = ap.parse_args()

    ensure_topics()
    prod = Producer({"bootstrap.servers": config.KAFKA_BOOTSTRAP, "linger.ms": 50, "compression.type": "lz4"})
    header = pd.read_csv(a.csv, nrows=0).columns
    cols = [c for c in RAW_COLUMNS if c in header]
    missing = sorted(set(RAW_COLUMNS) - set(cols))
    if missing:
        print(f"note: columns not in file (will be null): {missing}")
    if ID_COL not in cols:
        raise SystemExit(f"CSV has no '{ID_COL}' column")

    sent, errors = 0, 0

    def on_delivery(err, _msg):
        nonlocal errors
        if err:
            errors += 1

    t0 = time.time()
    for chunk in pd.read_csv(a.csv, usecols=cols, chunksize=a.chunksize, low_memory=False):
        chunk = chunk[chunk[ID_COL].notna()]
        # The raw Lending Club file ends with two summary rows with non-numeric ids.
        chunk = chunk[chunk[ID_COL].astype(str).str.fullmatch(r"\d+")]
        if a.issued_before or a.issued_after:
            d = pd.to_datetime(chunk[DATE_COL], format="%b-%Y", errors="coerce")
            if a.issued_before:
                chunk = chunk[d < pd.Timestamp(a.issued_before + "-01")]
                d = d.loc[chunk.index]
            if a.issued_after:
                chunk = chunk[d >= pd.Timestamp(a.issued_after + "-01")]
        if a.live and LABEL_COL in chunk:
            chunk = chunk.drop(columns=[LABEL_COL])
        for rec in chunk.to_dict("records"):
            rec = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in rec.items()}
            rec[ID_COL] = str(rec[ID_COL])
            while True:
                try:
                    prod.produce(config.TOPIC_APPLICATIONS, key=rec[ID_COL], value=json.dumps(rec),
                                 on_delivery=on_delivery)
                    break
                except BufferError:
                    prod.poll(0.5)
            prod.poll(0)
            sent += 1
            if a.rate:
                time.sleep(1 / a.rate)
            if sent % 10_000 == 0:
                print(f"  sent {sent:,}")
            if a.limit and sent >= a.limit:
                break
        if a.limit and sent >= a.limit:
            break
    prod.flush()
    print(f"done: {sent:,} messages to '{config.TOPIC_APPLICATIONS}' in {time.time() - t0:.1f}s, {errors} delivery errors")


if __name__ == "__main__":
    main()
