"""Central configuration. Everything is overridable via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_APPLICATIONS = os.getenv("TOPIC_APPLICATIONS", "loan-applications")
TOPIC_REVIEW = os.getenv("TOPIC_REVIEW", "underwriting-review")

WEBHDFS_URL = os.getenv("WEBHDFS_URL", "http://localhost:9870/webhdfs/v1")
HDFS_USER = os.getenv("HDFS_USER", "root")
HDFS_RAW_DIR = os.getenv("HDFS_RAW_DIR", "/lending_club/raw")

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ROOT / "models" / "metrics.json"))
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "scored.db"))

# Applications scoring strictly above this are sent to manual underwriting.
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.5"))
