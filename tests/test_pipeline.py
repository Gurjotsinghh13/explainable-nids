"""End-to-end smoke test on synthetic CICIDS2017-shaped data.

Checks preprocessing, the within-fold SMOTE stacking model (including that it
pickles), SHAP, and every API endpoint.  Run:  python -m pytest -q
"""

import os
import tempfile

import numpy as np
import pandas as pd

TMP = tempfile.mkdtemp()
os.environ["NIDS_ROOT"] = TMP

import joblib  # noqa: E402

from src import config  # noqa: E402
from src.evaluation import classification_metrics  # noqa: E402
from src.explainability import build_explainer, explain_one  # noqa: E402
from src.models import build_stacking_ensemble  # noqa: E402
from src.preprocessing import preprocess_inference, preprocess_training  # noqa: E402

LABELS = ["BENIGN"] * 1400 + ["Bot"] * 120 + ["DDoS"] * 300 + ["PortScan"] * 160 + \
         ["Web Attack \x96 XSS"] * 50 + ["Web Attack \x96 Brute Force"] * 40 + ["Heartbleed"] * 5


def synthetic_frame(seed=0):
    rng = np.random.default_rng(seed)
    n = len(LABELS)
    codes = pd.Series(LABELS).astype("category").cat.codes.to_numpy().astype(np.int64)
    df = pd.DataFrame({
        "Destination Port": rng.integers(0, 65535, n),
        "Flow Duration": rng.random(n) * 1e6 + codes * 2e5,
        "Flow Bytes/s": rng.random(n) * 5e3 + codes * 900,
        "Flow Packets/s": rng.random(n) * 100,
        "Init_Win_bytes_forward": rng.integers(0, 65535, n) // (codes + 1),
        "URG Flag Count": (codes % 2).astype(float),
        "Bwd Avg Bulk Rate": np.zeros(n),          # constant -> dropped
    })
    df["Fwd Header Length.1"] = df["Flow Duration"]  # duplicate -> dropped
    df.loc[3, "Flow Bytes/s"] = np.inf               # inf row -> dropped
    df[" Label"] = LABELS
    df.columns = df.columns.str.strip()
    return df


def test_end_to_end():
    d = preprocess_training(synthetic_frame(), verbose=False)
    assert "Destination Port" not in d["feature_names"]
    assert "Bwd Avg Bulk Rate" not in d["feature_names"]
    assert "Fwd Header Length.1" not in d["feature_names"]
    assert "Heartbleed" not in d["label_classes"] and "Web Attack" in d["label_classes"]
    assert d["X_train"].min() >= 0 and d["X_train"].max() <= 1

    stack = build_stacking_ensemble(floor=600, n_jobs=1, cv_folds=3)
    stack.fit(d["X_train"], d["y_train"])
    proba = stack.predict_proba(d["X_test"])
    m = classification_metrics(d["y_test"], proba.argmax(1), d["label_classes"], proba)
    assert m["macro_f1"] > 0.3

    for p in (config.PROCESSED_DIR, config.MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)
    joblib.dump(stack, config.MODEL_FILE)                       # must pickle
    joblib.dump(d["scaler"], config.SCALER_FILE)
    joblib.dump({k: d[k] for k in ("feature_names", "label_classes")}, config.PREPROCESSED_FILE)
    reloaded = joblib.load(config.MODEL_FILE)
    assert np.array_equal(reloaded.predict(d["X_test"]), stack.predict(d["X_test"]))

    top = explain_one(build_explainer(stack), d["X_test"][:1], 0, d["feature_names"], 3)
    assert len(top) == 3

    raw = synthetic_frame(1).drop(columns=["Label"]).iloc[:1]
    assert preprocess_inference(raw, d["scaler"], d["feature_names"]).shape == (1, len(d["feature_names"]))

    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        flow = {"features": {k: float(v) for k, v in raw.iloc[0].items()}, "true_label": "BENIGN"}
        r = client.post("/predict", json=flow).json()
        assert r["predicted_label"] in d["label_classes"]
        assert len(client.post("/explain", json=flow).json()["top_features"]) > 0
        assert client.get("/stats").json()["total_alerts"] == 1
        assert len(client.get("/alerts").json()) == 1
