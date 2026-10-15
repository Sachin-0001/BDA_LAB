"""Underwriting dashboard. `python -m lrisk.dashboard.app` -> http://localhost:5055"""
import json
import os

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from .. import config, store
from ..features import FEATURE_LABELS

app = Flask(__name__)

STATUSES = ("pending", "approved", "declined", "more_info", "all")
BINS = 20


def _metrics():
    try:
        return json.loads(config.METRICS_PATH.read_text())
    except (OSError, ValueError):
        return None


def _importance(metrics, top=8):
    gain = (metrics or {}).get("feature_importance_gain") or {}
    items = sorted(((v, k) for k, v in gain.items() if v > 0), reverse=True)[:top]
    peak = items[0][0] if items else 1
    return [{"label": FEATURE_LABELS.get(k, k), "pct": round(100 * v / peak, 1)} for v, k in items]


def _bars(hist):
    peak = max(hist) or 1
    return [{"lo": i / len(hist), "hi": (i + 1) / len(hist), "n": n, "pct": round(100 * n / peak, 1),
             "flag": i / len(hist) >= config.REVIEW_THRESHOLD} for i, n in enumerate(hist)]


def _wants_json():
    return request.accept_mimetypes.best == "application/json"


@app.get("/")
def index():
    status = request.args.get("status", "pending")
    if status not in STATUSES:
        status = "pending"
    s = store.stats(bins=BINS)
    metrics = _metrics()
    return render_template("index.html", stats=s, bars=_bars(s["histogram"]), rows=store.list_flagged(status),
                           status=status, statuses=STATUSES, threshold=config.REVIEW_THRESHOLD,
                           metrics=metrics, importance=_importance(metrics))


@app.post("/review/<application_id>")
def review(application_id):
    try:
        ok = store.decide(application_id, request.form["decision"], request.form.get("note", "").strip())
    except (KeyError, ValueError):
        abort(400)
    if _wants_json():
        return jsonify({"ok": ok, "stats": store.stats(bins=BINS)}), (200 if ok else 404)
    return redirect(url_for("index", status=request.args.get("status", "pending")))


@app.get("/api/stats")
def api_stats():
    return jsonify(store.stats(bins=BINS))


@app.get("/api/flagged")
def api_flagged():
    return jsonify(store.list_flagged(request.args.get("status", "pending"),
                                      int(request.args.get("limit", 200))))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5055")))
