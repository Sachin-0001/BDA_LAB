"""SQLite store shared by the scorer (writer) and the dashboard (reader / decision writer)."""
import json
import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    application_id TEXT PRIMARY KEY,
    scored_at      TEXT NOT NULL DEFAULT (datetime('now')),
    loan_amnt REAL, annual_inc REAL, dti REAL, fico REAL, term TEXT, purpose TEXT,
    score          REAL NOT NULL,
    flagged        INTEGER NOT NULL,
    reasons        TEXT,
    review_status  TEXT NOT NULL,       -- pending | approved | declined | more_info | not_required
    reviewed_at    TEXT,
    reviewer_note  TEXT
);
CREATE INDEX IF NOT EXISTS idx_scores_flag ON scores (flagged, review_status, score DESC);
"""

DECISIONS = ("approved", "declined", "more_info", "pending")


@contextmanager
def connect(path=None):
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert_scores(rows: list[dict], path=None) -> None:
    with connect(path) as con:
        con.executemany(
            """INSERT INTO scores (application_id, loan_amnt, annual_inc, dti, fico, term, purpose,
                                   score, flagged, reasons, review_status)
               VALUES (:application_id, :loan_amnt, :annual_inc, :dti, :fico, :term, :purpose,
                       :score, :flagged, :reasons, :review_status)
               ON CONFLICT(application_id) DO UPDATE SET score=excluded.score, flagged=excluded.flagged,
                   reasons=excluded.reasons, scored_at=datetime('now')""",
            [{**r, "reasons": json.dumps(r["reasons"])} for r in rows])


def decide(application_id: str, decision: str, note: str = "", path=None) -> bool:
    if decision not in DECISIONS:
        raise ValueError(decision)
    with connect(path) as con:
        cur = con.execute(
            "UPDATE scores SET review_status=?, reviewer_note=?, reviewed_at=datetime('now') "
            "WHERE application_id=? AND flagged=1", (decision, note, application_id))
        return cur.rowcount == 1


def stats(path=None, bins: int = 10) -> dict:
    with connect(path) as con:
        total, flagged, avg = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(flagged),0), COALESCE(AVG(score),0) FROM scores").fetchone()
        by_status = {r[0]: r[1] for r in con.execute(
            "SELECT review_status, COUNT(*) FROM scores WHERE flagged=1 GROUP BY review_status")}
        hist = [0] * bins
        for (s,) in con.execute("SELECT score FROM scores"):
            hist[min(int(s * bins), bins - 1)] += 1
    return {"total": total, "flagged": flagged, "avg_score": avg,
            "flag_rate": (flagged / total) if total else 0.0, "by_status": by_status, "histogram": hist}


def list_flagged(status: str | None = "pending", limit: int = 200, path=None) -> list[dict]:
    q, args = "SELECT * FROM scores WHERE flagged=1", []
    if status and status != "all":
        q += " AND review_status=?"
        args.append(status)
    q += " ORDER BY score DESC LIMIT ?"
    args.append(limit)
    with connect(path) as con:
        out = []
        for r in con.execute(q, args):
            d = dict(r)
            d["reasons"] = json.loads(d["reasons"] or "[]")
            out.append(d)
        return out
