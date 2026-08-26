"""FastAPI inference backend for the Explainable NIDS.

Loads the trained stacking model, the saved scaler, and a SHAP TreeExplainer
(built on the XGBoost base learner) once at startup. Exposes:

    GET  /health          liveness + what's loaded
    POST /predict         classify one raw flow, log the alert to SQLite
    POST /explain         per-alert SHAP explanation for one raw flow
    GET  /alerts          recent alerts from the log
    GET  /stats           class distribution + average latency

Run from the project root:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- make the project root importable so `src` works when run via uvicorn ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import preprocess_inference  # noqa: E402

MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "db" / "nids.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Artefacts loaded at startup
# --------------------------------------------------------------------------- #
STATE: dict = {}


def _extract_xgb(model):
    """Pull the fitted XGBoost base learner out of the stacking ensemble."""
    return model.named_estimators_["xgb"].named_steps["xgb"]


def load_artifacts():
    model = joblib.load(MODELS_DIR / "stacking_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    data = joblib.load(ROOT / "data" / "processed" / "preprocessed.joblib")

    xgb = _extract_xgb(model)
    try:
        xgb.set_params(device="cpu")   # SHAP/inference on CPU in the service
    except Exception:
        pass

    STATE.update(
        model=model,
        scaler=scaler,
        feature_names=data["feature_names"],
        class_names=data["label_classes"],
        explainer=shap.TreeExplainer(xgb),
    )


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS alerts (
               id            INTEGER PRIMARY KEY AUTOINCREMENT,
               timestamp     TEXT,
               predicted     TEXT,
               confidence    REAL,
               inference_ms  REAL,
               true_label    TEXT
           )"""
    )
    con.commit()
    con.close()


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class FlowRequest(BaseModel):
    features: Dict[str, float]
    true_label: Optional[str] = None


app = FastAPI(title="Explainable NIDS API", version="1.0.0")


@app.on_event("startup")
def _startup():
    load_artifacts()
    init_db()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_X(features: Dict[str, float]) -> np.ndarray:
    df = pd.DataFrame([features])
    return preprocess_inference(df, STATE["scaler"], STATE["feature_names"])


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    ready = "model" in STATE
    return {"status": "ok" if ready else "loading",
            "classes": STATE.get("class_names", []),
            "n_features": len(STATE.get("feature_names", []))}


@app.post("/predict")
def predict(req: FlowRequest):
    if "model" not in STATE:
        raise HTTPException(503, "Model not loaded")
    X = _to_X(req.features)

    t0 = time.perf_counter()
    proba = STATE["model"].predict_proba(X)[0]
    infer_ms = (time.perf_counter() - t0) * 1000.0

    idx = int(np.argmax(proba))
    label = STATE["class_names"][idx]
    confidence = float(proba[idx])

    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "INSERT INTO alerts (timestamp, predicted, confidence, inference_ms, true_label)"
        " VALUES (?,?,?,?,?)",
        (datetime.utcnow().isoformat(timespec="seconds"), label,
         confidence, infer_ms, req.true_label),
    )
    con.commit()
    alert_id = cur.lastrowid
    con.close()

    return {"alert_id": alert_id, "predicted_label": label,
            "confidence": round(confidence, 4), "inference_ms": round(infer_ms, 2)}


@app.post("/explain")
def explain(req: FlowRequest, top_k: int = 8):
    if "model" not in STATE:
        raise HTTPException(503, "Model not loaded")
    X = _to_X(req.features)

    idx = int(np.argmax(STATE["model"].predict_proba(X)[0]))   # class the stack chose
    exp = STATE["explainer"](X)                                # (1, n_feat, n_classes)
    vals = np.asarray(exp.values)[0, :, idx]

    order = np.argsort(np.abs(vals))[::-1][:top_k]
    feats = [
        {"feature": STATE["feature_names"][i],
         "shap_value": round(float(vals[i]), 4),
         "direction": "increases" if vals[i] > 0 else "decreases"}
        for i in order
    ]
    return {"predicted_label": STATE["class_names"][idx], "top_features": feats}


@app.get("/alerts")
def alerts(limit: int = 50):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.get("/stats")
def stats():
    con = sqlite3.connect(DB_PATH)
    dist = con.execute(
        "SELECT predicted, COUNT(*) n FROM alerts GROUP BY predicted ORDER BY n DESC"
    ).fetchall()
    avg = con.execute("SELECT AVG(inference_ms) FROM alerts").fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
    con.close()
    return {"total_alerts": total,
            "avg_inference_ms": round(avg, 2) if avg else None,
            "distribution": {p: n for p, n in dist}}