"""Underwriting dashboard. `python -m lrisk.dashboard.app` -> http://localhost:5055"""
import json
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for

from .. import config, store

app = Flask(__name__)


def _metrics():
    try:
        return json.loads(config.METRICS_PATH.read_text())
    except (OSError, ValueError):
        return None


@app.get("/")
def index():
    status = request.args.get("status", "pending")
    s = store.stats()
    peak = max(s["histogram"]) or 1
    bars = [{"lo": i / len(s["histogram"]), "n": n, "pct": round(100 * n / peak),
             "flag": i / len(s["histogram"]) >= config.REVIEW_THRESHOLD}
            for i, n in enumerate(s["histogram"])]
    return render_template("index.html", stats=s, bars=bars, rows=store.list_flagged(status),
                           status=status, threshold=config.REVIEW_THRESHOLD, metrics=_metrics())


@app.post("/review/<application_id>")
def review(application_id):
    store.decide(application_id, request.form["decision"], request.form.get("note", ""))
    return redirect(url_for("index", status=request.args.get("status", "pending")))


@app.get("/api/stats")
def api_stats():
    return jsonify(store.stats())


@app.get("/api/flagged")
def api_flagged():
    return jsonify(store.list_flagged(request.args.get("status", "pending"),
                                      int(request.args.get("limit", 200))))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5055")))
