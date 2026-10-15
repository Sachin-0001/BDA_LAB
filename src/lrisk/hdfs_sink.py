"""Kafka -> HDFS sink. Micro-batches messages into JSON-lines files on HDFS.

Layout: /lending_club/raw/ingest_date=YYYY-MM-DD/part<partition>-<firstOffset>-<lastOffset>.jsonl
Offsets are committed only after the batch is durably written (at-least-once). File names are
derived from offsets, so replays overwrite rather than duplicate; the trainer also de-duplicates
by application id.

    python -m lrisk.hdfs_sink                 # run forever
    python -m lrisk.hdfs_sink --idle-exit 15  # stop after 15s with no new messages
"""
import argparse
import datetime as dt
import time
from collections import defaultdict

from confluent_kafka import Consumer, KafkaError, TopicPartition

from . import config
from .hdfs_client import HDFS
from .kafka_utils import ensure_topics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--batch-seconds", type=float, default=5.0)
    ap.add_argument("--idle-exit", type=float, default=0, help="exit after N idle seconds (0 = never)")
    a = ap.parse_args()

    ensure_topics()
    hdfs = HDFS()
    c = Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP, "group.id": "hdfs-sink",
        "enable.auto.commit": False, "auto.offset.reset": "earliest",
    })
    c.subscribe([config.TOPIC_APPLICATIONS])

    buf: dict[int, list] = defaultdict(list)   # partition -> [(offset, bytes)]
    total, last_flush, last_msg = 0, time.time(), time.time()

    def flush() -> None:
        nonlocal total, last_flush
        day = dt.date.today().isoformat()
        offsets = []
        for part, items in buf.items():
            if not items:
                continue
            first, last = items[0][0], items[-1][0]
            path = f"{config.HDFS_RAW_DIR}/ingest_date={day}/part{part}-{first:012d}-{last:012d}.jsonl"
            hdfs.write(path, b"\n".join(v for _, v in items) + b"\n")
            offsets.append(TopicPartition(config.TOPIC_APPLICATIONS, part, last + 1))
            total += len(items)
            print(f"wrote {len(items):>5} msgs -> {path}")
        if offsets:
            c.commit(offsets=offsets, asynchronous=False)
        buf.clear()
        last_flush = time.time()

    try:
        while True:
            msg = c.poll(1.0)
            now = time.time()
            if msg is None:
                if buf and now - last_flush >= a.batch_seconds:
                    flush()
                if a.idle_exit and now - last_msg >= a.idle_exit:
                    break
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    raise RuntimeError(msg.error())
                continue
            last_msg = now
            buf[msg.partition()].append((msg.offset(), msg.value()))
            if sum(len(v) for v in buf.values()) >= a.batch_size or now - last_flush >= a.batch_seconds:
                flush()
    except KeyboardInterrupt:
        pass
    finally:
        flush()
        c.close()
        print(f"sink stopped; {total:,} messages persisted to HDFS")


if __name__ == "__main__":
    main()
