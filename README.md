# Module 4 — Prediction Sandbox

A stakeholder-facing Streamlit dashboard for the LightGBM ablation model (`g5`)
that answers the question: **"The model says N% click probability — *why*?"**

## What's in this folder

```
app.py                              # the dashboard (single file, ~600 lines)
requirements.txt                    # Python dependencies
best_ablation_model_g5.joblib       # trained LightGBM model (sklearn-wrapped)
best_ablation_model_g5.txt          # same model in LightGBM native format (fallback)
shap_background_sample.parquet      # 300-row background for SHAP
shap_background_sample.csv          # same, as CSV
README.md                           # this file
```

The app expects all of the above to sit in **the same folder as `app.py`**.
If yours live elsewhere, edit the paths at the top of `app.py` (the `CONFIG`
section — first block after the imports).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

> **Note on model loading** — the app tries `joblib` first and silently falls
> back to `lightgbm.Booster(model_file=…)` reading the `.txt` file if joblib
> fails (e.g. version mismatch between whatever saved the `.joblib` and your
> installed LightGBM). The active source is shown at the bottom of the
> sidebar.

## What the dashboard does

**Hero** — a live CTR gauge, three metrics (predicted CTR, baseline CTR
for a median user, total SHAP log-odds shift), and an auto-generated
plain-English narrative picking the top positive and negative drivers.

**Sidebar** — every one of the 27 model features exposed as a slider or
selectbox, grouped into five expanders (demographics, ad attributes,
behavior history, interactions, temporal). Two quick actions:
- **🎲 Random user** — pulls a row from the background sample and fills
  every widget at once.
- **↺ Reset to median** — returns to the median-user, median-ad baseline.

**Three detail tabs** —

1. **🔍 This prediction (local)** — SHAP waterfall plot for the current
   input plus a sortable table of all 27 feature contributions.
2. **🌐 Global importance** — mean |SHAP| across the background sample.
   A good cross-check against your Module 2 EDA: do the model's top
   drivers line up with the relationships you found in the data?
3. **📋 Raw input** — the exact 27-column row being fed to
   `model.predict()` right now. Useful when debugging scenarios.

## A couple of modelling notes worth knowing

- SHAP values and the base value are in **log-odds space**, not probability.
  The narrative quotes them that way ("+0.30 log-odds") because summing
  probability-space SHAP doesn't reconstruct the prediction cleanly. The
  gauge and metrics are always in probability.
- **`is_weekend`** and **`price_bucket`** are near-constant in this sample
  (all 1 and all 0 respectively), so the model hasn't really learned a
  response to them — nudging those sliders may produce a flat SHAP.
- **`pid`** (placement) only has two values in the training data, so its
  slider covers a very tight range.

## Extending the dashboard

The feature metadata lives in four dicts at the top of `app.py`:
`LABEL`, `CATEGORICAL`, `RANGES`, and `NARRATIVE`. To add or rename
features for a future ablation, edit those plus `FEATURE_ORDER` and
`GROUPS` — nothing else should need to change.
