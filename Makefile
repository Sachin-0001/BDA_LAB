PY      := PYTHONPATH=src .venv/bin/python
CSV     ?= data/sample_lending_club.csv
SPLIT   ?= 2017-01
LIVE_N  ?= 3000

.PHONY: help up down reset sample load train live dashboard test demo
help:
	@echo "make up        start Kafka + HDFS (docker)"
	@echo "make reset     wipe HDFS data, Kafka topics, scores DB and model (use before switching datasets)"
	@echo "make sample    generate SYNTHETIC data (only if you don't have the Kaggle file)"
	@echo "make load      CSV=<file>  stream history (issued before $(SPLIT)) -> Kafka -> HDFS"
	@echo "make train     train on data stored in HDFS"
	@echo "make live      run the scorer, then stream unlabeled 2017+ applications"
	@echo "make dashboard open http://localhost:5055"
	@echo "make demo      sample + load + train + live (synthetic data, end to end)"

up:
	docker compose up -d
	@until curl -sf "http://localhost:9870/webhdfs/v1/?op=LISTSTATUS&user.name=root" >/dev/null; do sleep 2; done; echo "HDFS ready"
	@until docker exec lrisk-kafka /opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092 >/dev/null 2>&1; do sleep 2; done; echo "Kafka ready"
down:
	docker compose down
reset:
	docker compose down -v
	rm -f data/scored.db data/scored.db-shm data/scored.db-wal models/model.joblib models/metrics.json
	$(MAKE) up
sample:
	$(PY) -m lrisk.generate_sample --rows 60000 --out data/sample_lending_club.csv
load:
	$(PY) -m lrisk.producer $(CSV) --issued-before $(SPLIT)
	$(PY) -m lrisk.hdfs_sink --idle-exit 10
train:
	$(PY) -m lrisk.train
live:
	( $(PY) -m lrisk.scorer --idle-exit 20 & ) ; sleep 8
	$(PY) -m lrisk.producer $(CSV) --issued-after $(SPLIT) --live --limit $(LIVE_N) --rate 100
	$(PY) -m lrisk.hdfs_sink --idle-exit 10
dashboard:
	$(PY) -m lrisk.dashboard.app
test:
	$(PY) -m pytest -q tests
demo: up sample load train live
	@echo "Done. Run: make dashboard"
