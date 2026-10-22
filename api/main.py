"""FastAPI inference backend for the Explainable NIDS.

Loads the stacking model, the saved scaler and a SHAP TreeExplainer once at
startup and exposes:

    GET  /health    liveness + what is loaded
    POST /predict   classify one raw flow and log it to SQLite
    POST /explain   per-alert SHAP explanation for one raw flow
    GET  /alerts    most recent logged alerts
    GET  /stats     class distribution and average latency

Run from the project root:
    python -m uvicorn api.main:app
"""

from __future__ import annotations

import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DB_PATH, MODEL_FILE, PREPROCESSED_FILE, SCALER_FILE  # noqa: E402
from src.explainability import build_explainer, explain_one  # noqa: E402
from src.preprocessing import preprocess_inference  # noqa: E402

STATE: dict = {}


def load_artifacts():
    model = joblib.load(MODEL_FILE)
    data = joblib.load(PREPROCESSED_FILE)
    STATE.update(model=model,
                 scaler=joblib.load(SCALER_FILE),
                 feature_names=data["feature_names"],
                 class_names=data["label_classes"],
                 explainer=build_explainer(model))


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS alerts (
                           id           INTEGER PRIMARY KEY AUTOINCREMENT,
                           timestamp    TEXT,
                           predicted    TEXT,
                           confidence   REAL,
                           inference_ms REAL,
                           true_label   TEXT)""")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_artifacts()
    init_db()
    yield


app = FastAPI(title="Explainable NIDS API", version="1.0.0", lifespan=lifespan)


class FlowRequest(BaseModel):
    features: Dict[str, float]
    true_label: Optional[str] = None


def _to_X(features: Dict[str, float]) -> np.ndarray:
    return preprocess_inference(pd.DataFrame([features]), STATE["scaler"], STATE["feature_names"])


def _require_model():
    if "model" not in STATE:
        raise HTTPException(503, "Model not loaded")


@app.get("/health")
def health():
    return {"status": "ok" if "model" in STATE else "loading",
            "classes": STATE.get("class_names", []),
            "n_features": len(STATE.get("feature_names", []))}


@app.post("/predict")
def predict(req: FlowRequest):
    _require_model()
    X = _to_X(req.features)
    t0 = time.perf_counter()
    proba = STATE["model"].predict_proba(X)[0]
    infer_ms = (time.perf_counter() - t0) * 1000.0
    idx = int(np.argmax(proba))
    label, conf = STATE["class_names"][idx], float(proba[idx])

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            "INSERT INTO alerts (timestamp, predicted, confidence, inference_ms, true_label)"
            " VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), label, conf,
             infer_ms, req.true_label))
        alert_id = cur.lastrowid
    return {"alert_id": alert_id, "predicted_label": label,
            "confidence": round(conf, 4), "inference_ms": round(infer_ms, 2)}


@app.post("/explain")
def explain(req: FlowRequest, top_k: int = 8):
    _require_model()
    X = _to_X(req.features)
    idx = int(np.argmax(STATE["model"].predict_proba(X)[0]))
    feats = explain_one(STATE["explainer"], X, idx, STATE["feature_names"], top_k)
    return {"predicted_label": STATE["class_names"][idx], "top_features": feats}


@app.get("/alerts")
def alerts(limit: int = 50):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/stats")
def stats():
    with sqlite3.connect(DB_PATH) as con:
        dist = con.execute("SELECT predicted, COUNT(*) FROM alerts GROUP BY predicted"
                           " ORDER BY COUNT(*) DESC").fetchall()
        avg = con.execute("SELECT AVG(inference_ms) FROM alerts").fetchone()[0]
        total = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    return {"total_alerts": total,
            "avg_inference_ms": round(avg, 2) if avg else None,
            "distribution": {p: n for p, n in dist}}
