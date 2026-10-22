"""End-to-end training: load -> preprocess -> baselines -> stacking -> evaluate -> save.

Usage (from the project root):
    python -m scripts.train                       # full run (~1 h on a laptop CPU)
    python -m scripts.train --device cuda         # XGBoost on an NVIDIA GPU
    python -m scripts.train --skip-baselines --skip-cv
    python -m scripts.train --data-dir path/to/MachineLearningCVE
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np

from src.config import (ENCODER_FILE, FIGURES_DIR, MODEL_FILE, MODELS_DIR,
                        PREPROCESSED_FILE, PROCESSED_DIR, REPORTS_DIR,
                        SCALER_FILE, SMOTE_FLOOR)
from src.data_loading import load_all_csvs
from src.evaluation import (class_recall, classification_metrics,
                            cross_validated_stability,
                            measure_inference_latency, measure_shap_overhead)
from src.explainability import build_explainer, global_importance
from src.models import build_random_forest, build_stacking_ensemble, build_xgboost
from src.preprocessing import preprocess_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (XGBoost)")
    ap.add_argument("--n-jobs", type=int, default=4, help="Random Forest parallelism")
    ap.add_argument("--floor", type=int, default=SMOTE_FLOOR)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()

    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1-2. Load and preprocess (or reuse the cached split)
    if PREPROCESSED_FILE.exists():
        print("Loading cached preprocessed data:", PREPROCESSED_FILE)
        d = joblib.load(PREPROCESSED_FILE)
        scaler, le = joblib.load(SCALER_FILE), joblib.load(ENCODER_FILE)
    else:
        d = preprocess_training(load_all_csvs(args.data_dir))
        scaler, le = d.pop("scaler"), d.pop("label_encoder")
        joblib.dump(d, PREPROCESSED_FILE, compress=3)
        joblib.dump(scaler, SCALER_FILE)
        joblib.dump(le, ENCODER_FILE)
    X_tr, X_te, y_tr, y_te = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
    classes, feats = d["label_classes"], d["feature_names"]
    bot = classes.index("Bot") if "Bot" in classes else None
    print(f"Train {X_tr.shape}  Test {X_te.shape}  Classes {len(classes)}")

    results = {"comparison": {}}

    def summarise(name, pred, proba=None):
        m = classification_metrics(y_te, pred, classes, proba)
        row = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "weighted_f1": m["weighted_f1"], "macro_fpr": m["macro_fpr"]}
        if bot is not None:
            row["bot_recall"] = class_recall(y_te, pred, bot)
        results["comparison"][name] = row
        print(f"{name:32s} acc={row['accuracy']:.4f} macroF1={row['macro_f1']:.4f}")
        return m

    # 3. Baselines (no resampling)
    if not args.skip_baselines:
        rf = build_random_forest(args.n_jobs).fit(X_tr, y_tr)
        summarise("RandomForest (baseline)", rf.predict(X_te))
        xgb = build_xgboost(args.device).fit(X_tr, y_tr)
        summarise("XGBoost (baseline)", xgb.predict(X_te))

    # 4. Stacking ensemble with within-fold SMOTE
    t0 = time.time()
    stack = build_stacking_ensemble(floor=args.floor, n_jobs=args.n_jobs, device=args.device)
    stack.fit(X_tr, y_tr)
    print(f"Stacking trained in {(time.time() - t0) / 60:.1f} min")
    joblib.dump(stack, MODEL_FILE, compress=3)

    proba = stack.predict_proba(X_te)
    m = summarise("Stacking + within-fold SMOTE", proba.argmax(1), proba)
    results["stacking"] = {"roc_auc_ovr": m["roc_auc_ovr"], "macro_fpr": m["macro_fpr"]}
    m["per_class"].round(4).to_csv(REPORTS_DIR / "stacking_per_class.csv")
    np.savetxt(REPORTS_DIR / "confusion_matrix.csv", m["confusion_matrix"], fmt="%d", delimiter=",")

    # 5. Explainability + timing
    explainer = build_explainer(stack)
    one = X_te[:1]
    results["latency_ms"] = measure_inference_latency(stack, one)
    results["shap_ms"] = measure_shap_overhead(explainer, one)
    results["global_importance"] = global_importance(explainer, X_te[:2000], feats)

    # 6. Cross-validated stability
    if not args.skip_cv:
        results["cv"] = cross_validated_stability(
            lambda floor: build_stacking_ensemble(floor=floor, n_jobs=2, device=args.device),
            X_tr, y_tr)

    with open(REPORTS_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("Saved model ->", MODEL_FILE)
    print("Saved results ->", REPORTS_DIR / "results.json")


if __name__ == "__main__":
    main()
