# -*- coding: utf-8 -*-
"""SQLite 存储：分析会话、学生情绪档案与预警记录（零依赖）。"""
import json
import sqlite3
import time

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,              -- interview | classroom
    engine TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS alerts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    subject TEXT,                    -- track_id 或 受访者
    level TEXT,
    reasons_json TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
"""


def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.executescript(_SCHEMA)
    return c


def save_session(mode, engine, summary, alerts):
    c = _conn()
    cur = c.execute("INSERT INTO sessions(mode, engine, summary_json) VALUES(?,?,?)",
                    (mode, engine, json.dumps(summary, ensure_ascii=False)))
    sid = cur.lastrowid
    for a in alerts:
        c.execute("INSERT INTO alerts(session_id, subject, level, reasons_json) "
                  "VALUES(?,?,?,?)",
                  (sid, str(a.get("track_id", "受访者")), a["level"],
                   json.dumps(a["reasons"], ensure_ascii=False)))
    c.commit(); c.close()
    return sid


def history(limit=50):
    c = _conn()
    rows = c.execute("SELECT id, mode, engine, created_at, summary_json "
                     "FROM sessions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = [{"id": r[0], "mode": r[1], "engine": r[2], "created_at": r[3],
            "summary": json.loads(r[4] or "{}")} for r in rows]
    c.close()
    return out


def alert_history(limit=100):
    c = _conn()
    rows = c.execute("SELECT session_id, subject, level, reasons_json, created_at "
                     "FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = [{"session_id": r[0], "subject": r[1], "level": r[2],
            "reasons": json.loads(r[3]), "created_at": r[4]} for r in rows]
    c.close()
    return out
