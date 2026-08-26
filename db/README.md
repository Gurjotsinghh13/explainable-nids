# Database (SQLite)

Lightweight storage for alert logs and inference history. SQLite ships with
Python's standard library — no extra install needed.

- The database file (`nids.db`) is created by the backend at runtime and is
  **gitignored**.
- Default path is set in `src/config.py` as `DB_PATH`.

## Planned schema (finalised during the backend step)

`alerts` — one row per flagged flow:
- id, timestamp, predicted_label, confidence, inference_ms, top_shap_features
