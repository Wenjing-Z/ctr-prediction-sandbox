"""
Module 4 — Prediction Sandbox
=========================================================================
Live CTR prediction + per-prediction SHAP explanation for the LightGBM
model with the engineered feature set described in schema_details.pdf.
Built to answer the stakeholder question:

    "The model says N% click probability — *why*?"

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit paths here if your files live elsewhere
# ═══════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).parent
MODEL_JOBLIB = ROOT / "models" / "lgbm_model.joblib"
MODEL_TXT    = ROOT / "models" / "lgbm_model.txt"   
BG_PARQUET   = ROOT / "data" / "shap_sample_data_lgbm.parquet"
BG_CSV       = ROOT / "data" / "shap_sample_data_lgbm.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE METADATA (30 features)
# ═══════════════════════════════════════════════════════════════════════════════
# Exact order the LightGBM model expects (taken from the .txt header).
FEATURE_ORDER = [
    "final_gender_code", "age_level", "occupation", "shopping_level",
    "cms_segid", "cms_group_id", "pvalue_level", "new_user_class_level",
    "hour", "weekday", "price",
    "ad_ctr_cv", "user_ctr_cv",
    "brand_te", "cate_id_te", "pid_te", "cms_group_id_te",
    "gender_cate", "age_cate", "brand_cate",
    "time_since_last_click",
    "user_count_1H", "user_ctr_1H",
    "user_count_1D", "user_ctr_1D",
    "user_count_3D", "user_ctr_3D",
    "affinity_score", "total_interactions", "interest_recency",
]

LABEL = {
    # Demographics
    "final_gender_code":     "Gender",
    "age_level":             "Age bracket",
    "occupation":            "Student",
    "shopping_level":        "Shopping depth",
    "cms_segid":             "Micro-segment ID",
    "cms_group_id":          "Demographic group",
    "pvalue_level":          "Consumption level",
    "new_user_class_level":  "City tier",
    # Temporal
    "hour":                  "Hour of day",
    "weekday":               "Day of week",
    # Ad
    "price":                 "Price (log1p)",
    "ad_ctr_cv":             "Ad historical CTR (5-fold CV)",
    "brand_te":              "Brand CTR (target-enc.)",
    "cate_id_te":            "Category CTR (target-enc.)",
    "pid_te":                "Placement CTR (target-enc.)",
    "cms_group_id_te":       "Demo-group CTR (target-enc.)",
    # User behaviour
    "user_ctr_cv":           "User historical CTR (5-fold CV)",
    "user_count_1H":         "Impressions last hour",
    "user_ctr_1H":           "CTR last hour",
    "user_count_1D":         "Impressions last day",
    "user_ctr_1D":           "CTR last day",
    "user_count_3D":         "Impressions last 3 days",
    "user_ctr_3D":           "CTR last 3 days",
    "time_since_last_click": "Seconds since last click",
    # Interaction clusters (integer cluster IDs — top 150 + 0=other)
    "gender_cate":           "Gender × Category cluster",
    "age_cate":              "Age × Category cluster",
    "brand_cate":            "Brand × Category cluster",
    # Category affinity
    "affinity_score":        "Category affinity score",
    "total_interactions":    "Total category interactions",
    "interest_recency":      "Time since last category interaction",
}

# Pure-categorical features → selectbox with labels
CATEGORICAL = {
    "final_gender_code":    {-1: "Unknown", 1: "Male", 2: "Female"},
    "age_level":            {-1: "Unknown", 1: "< 18", 2: "18–24", 3: "25–29",
                             4: "30–34", 5: "35–39", 6: "40+"},
    "occupation":           {-1: "Unknown", 0: "Not student", 1: "Student"},
    "shopping_level":       {-1: "Unknown", 1: "Shallow", 2: "Moderate", 3: "Deep"},
    "pvalue_level":         {-1: "Unknown", 1: "Low", 2: "Mid", 3: "High"},
    "new_user_class_level": {-1: "Unknown", 1: "Tier 1", 2: "Tier 2",
                             3: "Tier 3", 4: "Tier 4"},
    "weekday":              {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                             3: "Thursday", 4: "Friday", 5: "Saturday",
                             6: "Sunday"},
}

# (min, max, step) — pulled from LightGBM feature_infos in the .txt header.
# Slightly widened where the background sample exceeds the training range.
RANGES = {
    "cms_segid":             (-1,        96,           1),
    "cms_group_id":          (-1,        12,           1),
    "hour":                  (0,         23,           1),
    "price":                 (0.0,       16.6,         0.1),
    "ad_ctr_cv":             (0.0,       1.0,          0.01),
    "user_ctr_cv":           (0.0,       1.0,          0.01),
    "brand_te":              (0.0,       1.0,          0.01),
    "cate_id_te":            (0.0,       1.0,          0.01),
    "pid_te":                (0.0508,    0.0538,       0.0001),
    "cms_group_id_te":       (0.0,       0.06,         0.001),
    "gender_cate":           (0,         150,          1),
    "age_cate":              (0,         150,          1),
    "brand_cate":            (0,         150,          1),
    "time_since_last_click": (0,         999_999,      60),
    "user_count_1H":         (0,         18,           1),
    "user_ctr_1H":           (0.0,       1.0,          0.01),
    "user_count_1D":         (0,         47,           1),
    "user_ctr_1D":           (0.0,       1.0,          0.01),
    "user_count_3D":         (0,         91,           1),
    "user_ctr_3D":           (0.0,       1.0,          0.01),
    "affinity_score":        (0.0,       10_200.0,     1.0),
    "total_interactions":    (0,         10_038,       1),
    "interest_recency":      (1,         1_500_000_000, 60_000),
}

GROUPS = {
    "👤  User demographics": [
        "final_gender_code", "age_level", "pvalue_level", "shopping_level",
        "occupation", "new_user_class_level", "cms_segid", "cms_group_id",
    ],
    "🎯  Ad attributes": [
        "price", "ad_ctr_cv", "brand_te", "cate_id_te", "pid_te",
    ],
    "📈  User behaviour history": [
        "user_ctr_cv",
        "user_count_1H", "user_ctr_1H",
        "user_count_1D", "user_ctr_1D",
        "user_count_3D", "user_ctr_3D",
        "time_since_last_click",
    ],
    "🔗  Cross / interaction clusters": [
        "gender_cate", "age_cate", "brand_cate", "cms_group_id_te",
    ],
    "💝  Category affinity": [
        "affinity_score", "total_interactions", "interest_recency",
    ],
    "🕒  Temporal context": [
        "weekday", "hour",
    ],
}

# (phrase-when-SHAP-is-positive, phrase-when-SHAP-is-negative)
NARRATIVE = {
    # demographics
    "final_gender_code":     ("gender matches typical clickers",
                              "gender rarely clicks this kind of ad"),
    "age_level":             ("age bracket aligns with typical clickers",
                              "age bracket rarely clicks this kind of ad"),
    "occupation":            ("student status helps here",
                              "student status hurts here"),
    "shopping_level":        ("deep shopper — naturally engaged",
                              "shallow shopper — usually skims past"),
    "cms_segid":             ("micro-segment leans toward clicks",
                              "micro-segment leans away from clicks"),
    "cms_group_id":          ("demographic group leans toward clicks",
                              "demographic group leans away from clicks"),
    "pvalue_level":          ("consumption tier is a strong buyer profile",
                              "consumption tier rarely engages"),
    "new_user_class_level":  ("city tier is a hotspot for this ad",
                              "city tier rarely responds"),
    # temporal
    "hour":                  ("hour of day is prime-time for clicks",
                              "hour of day is a dead zone for clicks"),
    "weekday":               ("day of week tilts positive",
                              "day of week tilts negative"),
    # ad
    "price":                 ("price is attractive for this user",
                              "price looks high given the context"),
    "ad_ctr_cv":             ("this ad has a strong historical click record",
                              "this ad has a weak historical click record"),
    "brand_te":              ("brand has a strong click track record",
                              "brand has a weak click track record"),
    "cate_id_te":            ("ad's category has a historically high CTR",
                              "ad's category has a historically low CTR"),
    "pid_te":                ("placement slot performs well on average",
                              "placement slot performs poorly on average"),
    "cms_group_id_te":       ("demographic group's CTR profile boosts this prediction",
                              "demographic group's CTR profile hurts this prediction"),
    # behaviour
    "user_ctr_cv":           ("user is a historically engaged clicker",
                              "user historically rarely clicks"),
    "user_ctr_3D":           ("user has been clicking frequently in the last 3 days",
                              "user has been largely ignoring ads in the last 3 days"),
    "user_ctr_1D":           ("user is highly engaged today",
                              "user has been unengaged today"),
    "user_ctr_1H":           ("user is actively clicking right now",
                              "user has been quiet in the last hour"),
    "user_count_3D":         ("user has seen many ads recently (active browser)",
                              "user has been seen few ads recently (cold start)"),
    "user_count_1D":         ("user has been browsing heavily today",
                              "user has barely browsed today"),
    "user_count_1H":         ("user is active in-session",
                              "user is not currently active"),
    "time_since_last_click": ("user clicked recently (high intent)",
                              "user hasn't clicked in a long time"),
    # interaction clusters
    "gender_cate":           ("gender × category combo is a strong predictor here",
                              "gender × category combo is a weak predictor here"),
    "age_cate":              ("age × category combo is a strong predictor here",
                              "age × category combo is a weak predictor here"),
    "brand_cate":            ("brand × category combo is a strong predictor here",
                              "brand × category combo is a weak predictor here"),
    # affinity
    "affinity_score":        ("user has built strong affinity with this category",
                              "user has weak affinity with this category"),
    "total_interactions":    ("user has many past interactions in this category",
                              "user has few past interactions in this category"),
    "interest_recency":      ("recency of last category interaction boosts this",
                              "long gap since last category interaction hurts this"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE + CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CTR Sandbox · Module 4",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
      [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 600; }
      [data-testid="stSidebar"] { background: #fafafa; }
      [data-testid="stSidebar"] .stExpander { border: 1px solid #e5e7eb; border-radius: 8px; }
      h1, h2, h3 { letter-spacing: -0.015em; }
      .hero-title {
          font-family: 'Georgia', serif;
          font-size: 2.6rem; font-weight: 600;
          margin: 0; padding: 0; color: #111;
      }
      .hero-kicker {
          text-transform: uppercase; letter-spacing: 0.22em;
          font-size: 0.75rem; color: #6366f1; font-weight: 700;
          margin-bottom: 0.35rem;
      }
      .narrative {
          background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 100%);
          border-left: 4px solid #6366f1;
          padding: 1.1rem 1.3rem;
          border-radius: 10px;
          line-height: 1.65;
          font-size: 1.02rem;
      }
      .narrative ul { margin: 0.4rem 0 0 1.1rem; padding: 0; }
      .narrative li { margin: 0.25rem 0; }
      .chip-pos { color: #15803d; font-weight: 600; }
      .chip-neg { color: #b91c1c; font-weight: 600; }
      .tiny { color: #6b7280; font-size: 0.82rem; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _py(val, feat):
    """Cast a numpy/pandas scalar to a native Python value matching the
    widget type (int for integer sliders & categoricals, float otherwise),
    clipped into the feature's valid range."""
    if hasattr(val, "item"):
        val = val.item()
    if feat in CATEGORICAL:
        options = CATEGORICAL[feat]
        v = int(round(float(val)))
        return v if v in options else next(iter(options))
    step = RANGES.get(feat, (0, 0, 1))[2]
    lo, hi, _ = RANGES[feat]
    if isinstance(step, int):
        return max(int(lo), min(int(hi), int(round(float(val)))))
    return float(max(lo, min(hi, float(val))))

# ═══════════════════════════════════════════════════════════════════════════════
# LOADERS (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Prefer joblib if it exists; otherwise load LightGBM .txt."""
    if MODEL_JOBLIB.exists():
        try:
            return joblib.load(MODEL_JOBLIB), "joblib"
        except Exception as err:
            st.warning(f"joblib load failed ({type(err).__name__}); falling back to .txt")
    if MODEL_TXT.exists():
        import lightgbm as lgb
        return lgb.Booster(model_file=str(MODEL_TXT)), "lightgbm .txt"
    raise FileNotFoundError(
        f"No model file found. Expected {MODEL_JOBLIB.name} or {MODEL_TXT.name} "
        f"next to app.py."
    )

@st.cache_data(show_spinner="Loading background sample…")
def load_background() -> pd.DataFrame:
    if BG_PARQUET.exists():
        df = pd.read_parquet(BG_PARQUET)
    elif BG_CSV.exists():
        df = pd.read_csv(BG_CSV)
    else:
        raise FileNotFoundError(
            f"Need {BG_PARQUET.name} or {BG_CSV.name} next to app.py"
        )
    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        raise ValueError(
            f"Background sample is missing required columns: {missing}"
        )
    return df[FEATURE_ORDER].copy()

@st.cache_resource(show_spinner="Building SHAP explainer…")
def build_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data(show_spinner="Computing global importance…")
def global_importance(_explainer, bg: pd.DataFrame, n_sample: int = 500) -> pd.DataFrame:
    sample = bg.sample(min(n_sample, len(bg)), random_state=0)
    exp = _explainer(sample)
    vals = np.asarray(exp.values)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    mean_abs = np.abs(vals).mean(axis=0)
    return (
        pd.DataFrame({
            "feature":       FEATURE_ORDER,
            "label":         [LABEL[f] for f in FEATURE_ORDER],
            "mean_abs_shap": mean_abs,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT + EXPLAIN
# ═══════════════════════════════════════════════════════════════════════════════
def predict_ctr(model, row: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(row)[:, 1][0])
    return float(model.predict(row)[0])   # LightGBM Booster w/ binary objective

def explain_row(explainer, row: pd.DataFrame):
    """Return (shap_values shape (n_feat,), base_value) in log-odds space."""
    exp = explainer(row)
    vals = np.asarray(exp.values)
    base = np.asarray(exp.base_values)
    if vals.ndim == 3:                     # (1, n_feat, n_class)
        sv = vals[0, :, -1].astype(float)
        bv = float(base[0, -1]) if base.ndim >= 2 else float(base[-1])
    elif vals.ndim == 2:                   # (1, n_feat)
        sv = vals[0].astype(float)
        bv = float(base[0]) if base.ndim >= 1 else float(base)
    else:                                  # (n_feat,)
        sv = vals.astype(float)
        bv = float(base)
    return sv, bv

def generate_narrative(shap_vals, values, top_k=3, threshold=0.01):
    pairs = list(zip(FEATURE_ORDER, shap_vals))
    pos = sorted([p for p in pairs if p[1] >  threshold], key=lambda x: -x[1])[:top_k]
    neg = sorted([p for p in pairs if p[1] < -threshold], key=lambda x:  x[1])[:top_k]

    def phrase(feat, sv):
        if feat not in NARRATIVE:
            return f"{LABEL.get(feat, feat)} = {values[feat]}"
        p_pos, p_neg = NARRATIVE[feat]
        return p_pos if sv > 0 else p_neg

    return {
        "positive": [(f, sv, phrase(f, sv)) for f, sv in pos],
        "negative": [(f, sv, phrase(f, sv)) for f, sv in neg],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def gauge(prob: float) -> go.Figure:
    color = "#94a3b8" if prob < 0.02 else ("#f59e0b" if prob < 0.08 else "#22c55e")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 46, "family": "Georgia"}},
        gauge={
            "axis":    {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar":     {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   2], "color": "#f8fafc"},
                {"range": [2,   8], "color": "#fef3c7"},
                {"range": [8, 100], "color": "#dcfce7"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=230,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def waterfall_fig(shap_vals, base_value, feat_vals, max_display=12):
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=feat_vals,
        feature_names=[LABEL[f] for f in FEATURE_ORDER],
    )
    plt.figure(figsize=(8, 5.5))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    return plt.gcf()

# ═══════════════════════════════════════════════════════════════════════════════
# BOOT + SESSION-STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════
model, model_source = load_model()
bg = load_background()
explainer = build_explainer(model)
gimp = global_importance(explainer, bg)

# Baseline = column-wise median of background sample
BASELINE = {f: _py(bg[f].median(), f) for f in FEATURE_ORDER}

for f in FEATURE_ORDER:
    if f not in st.session_state:
        st.session_state[f] = BASELINE[f]

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — inputs
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### 🎛  Scenario controls")
st.sidebar.caption("Adjust any feature — prediction & SHAP update live.")

c1, c2 = st.sidebar.columns(2)
if c1.button("🎲  Random user", use_container_width=True):
    row = bg.sample(1).iloc[0]
    for f in FEATURE_ORDER:
        st.session_state[f] = _py(row[f], f)
    st.rerun()
if c2.button("↺  Reset to median", use_container_width=True):
    for f in FEATURE_ORDER:
        st.session_state[f] = BASELINE[f]
    st.rerun()

st.sidebar.markdown("---")

def render_widget(feat: str):
    label = LABEL[feat]
    if feat in CATEGORICAL:
        opts = list(CATEGORICAL[feat].keys())
        st.selectbox(
            label, opts,
            index=opts.index(st.session_state[feat]),
            format_func=lambda k: CATEGORICAL[feat][k],
            key=feat,
        )
        return
    lo, hi, step = RANGES[feat]
    if isinstance(step, int):
        st.slider(label, int(lo), int(hi), step=int(step), key=feat)
    else:
        fmt = "%.4f" if step < 0.01 else ("%.3f" if step < 0.1 else "%.2f")
        st.slider(label, float(lo), float(hi), step=float(step),
                  format=fmt, key=feat)

for group_title, feats in GROUPS.items():
    with st.sidebar.expander(group_title, expanded=group_title.startswith("👤")):
        for f in feats:
            render_widget(f)

st.sidebar.markdown("---")
st.sidebar.caption(f"Model source · `{model_source}`  ·  {len(FEATURE_ORDER)} features")

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD INPUT ROW + PREDICT + SHAP
# ═══════════════════════════════════════════════════════════════════════════════
current_vals = {f: st.session_state[f] for f in FEATURE_ORDER}
X_row  = pd.DataFrame([[current_vals[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
X_base = pd.DataFrame([[BASELINE[f]     for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

p_ctr  = predict_ctr(model, X_row)
p_base = predict_ctr(model, X_base)
shap_vals, base_lo = explain_row(explainer, X_row)

narrative = generate_narrative(shap_vals, current_vals)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-kicker">Module 4 · Tab 2 · Prediction Sandbox</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Why did the model predict this CTR?</h1>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="tiny">A live, stakeholder-facing explanation of one specific '
    'prediction — what the model thinks and which features got it there.</p>',
    unsafe_allow_html=True,
)
st.write("")

hero_l, hero_r = st.columns([0.42, 0.58], gap="large")

with hero_l:
    st.plotly_chart(gauge(p_ctr), use_container_width=True,
                    config={"displayModeBar": False})
    m1, m2, m3 = st.columns(3)
    delta_pp = (p_ctr - p_base) * 100
    m1.metric("Predicted CTR", f"{p_ctr*100:.2f}%",
              f"{delta_pp:+.2f} pp vs baseline")
    m2.metric("Baseline CTR", f"{p_base*100:.2f}%",
              help="Median-user, median-ad from the background sample")
    m3.metric("Σ SHAP", f"{shap_vals.sum():+.3f}",
              help="Total log-odds shift from the model's baseline")

with hero_r:
    pos, neg = narrative["positive"], narrative["negative"]
    parts = ['<div class="narrative">']
    parts.append(
        f"<b>The model predicts a {p_ctr*100:.1f}% click probability</b> "
        f"(versus a {p_base*100:.1f}% baseline). Top drivers:"
    )
    parts.append("<ul>")
    for f, sv, ph in pos:
        parts.append(
            f'<li><span class="chip-pos">↑ pushes up</span> — {ph} '
            f'<span class="tiny">({sv:+.3f} log-odds)</span></li>'
        )
    for f, sv, ph in neg:
        parts.append(
            f'<li><span class="chip-neg">↓ pulls down</span> — {ph} '
            f'<span class="tiny">({sv:+.3f} log-odds)</span></li>'
        )
    if not pos and not neg:
        parts.append(
            '<li>No single feature is swinging the prediction strongly — '
            'this profile looks typical.</li>'
        )
    parts.append("</ul></div>")
    st.markdown("\n".join(parts), unsafe_allow_html=True)

st.write("")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — DETAIL TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_local, tab_global, tab_data = st.tabs(
    ["🔍  This prediction (local)", "🌐  Global importance", "📋  Raw input"]
)

with tab_local:
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        st.markdown("#### SHAP waterfall")
        st.caption(
            "Each bar is a feature's push — in log-odds — from the model's "
            "baseline toward this prediction."
        )
        max_disp = st.slider("Show top N features", 5, 25, 12, 1,
                             key="wf_max_display")
        fig = waterfall_fig(shap_vals, base_lo,
                            X_row.iloc[0].values, max_display=max_disp)
        st.pyplot(fig, use_container_width=True, clear_figure=True)

    with right:
        st.markdown("#### Feature contributions")
        df_contrib = (
            pd.DataFrame({
                "Feature": [LABEL[f] for f in FEATURE_ORDER],
                "Value":   [current_vals[f] for f in FEATURE_ORDER],
                "SHAP":    shap_vals,
            })
            .sort_values("SHAP", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            df_contrib,
            use_container_width=True,
            height=480,
            column_config={
                "Value": st.column_config.NumberColumn(format="%.4f"),
                "SHAP":  st.column_config.NumberColumn(
                    "SHAP Δ (log-odds)", format="%+.3f"
                ),
            },
            hide_index=True,
        )

with tab_global:
    st.markdown("#### Mean |SHAP| across the background sample")
    st.caption(
        "Which features matter most to the model overall? Cross-check this "
        "against the Module 2 EDA — do the model's top drivers align with "
        "the patterns you found in the data?"
    )
    top_n = st.slider("Top N", 5, len(FEATURE_ORDER), 15, 1, key="g_top_n")
    g_view = gimp.head(top_n).iloc[::-1]
    fig_g = go.Figure(go.Bar(
        x=g_view["mean_abs_shap"],
        y=g_view["label"],
        orientation="h",
        marker=dict(color="#6366f1"),
        hovertemplate="<b>%{y}</b><br>mean |SHAP| = %{x:.4f}<extra></extra>",
    ))
    fig_g.update_layout(
        height=40 + 24 * top_n,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="mean |SHAP| (log-odds)",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_g, use_container_width=True,
                    config={"displayModeBar": False})

with tab_data:
    st.markdown("#### Current scenario — exact model input row")
    display_df = X_row.T.rename(columns={0: "value"})
    display_df["label"] = [LABEL[f] for f in FEATURE_ORDER]
    st.dataframe(
        display_df[["label", "value"]],
        use_container_width=True,
        height=520,
    )




# """
# Module 4 — Prediction Sandbox (Tab 2 of the CTR Explainability Dashboard)
# =========================================================================
# Live CTR prediction + per-prediction SHAP explanation for the LightGBM
# ablation model (g5). Built to answer the stakeholder question:

#     "The model says 80% click probability — why?"

# Run:
#     pip install -r requirements.txt
#     streamlit run app.py
# """
# from __future__ import annotations

# from pathlib import Path

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import shap
# import streamlit as st

# # ═══════════════════════════════════════════════════════════════════════════════
# # CONFIG — edit paths here if your files live elsewhere
# # ═══════════════════════════════════════════════════════════════════════════════
# ROOT = Path(__file__).parent
# MODEL_JOBLIB = ROOT / "lgbm_model.joblib"
# MODEL_TXT    = ROOT / "lgbm_model.txt"
# BG_PARQUET   = ROOT / "shap_sample_data_lgbm.parquet"
# BG_CSV       = ROOT / "shap_sample_data_lgbm.csv"

# # ═══════════════════════════════════════════════════════════════════════════════
# # FEATURE METADATA
# # ═══════════════════════════════════════════════════════════════════════════════
# # Exact order the LightGBM model expects (taken from the .txt header).

# Feature_order = [
#     'final_gender_code', 'age_level', 'occupation', 'shopping_level',
#     'cms_segid', 'cms_group_id', 'pvalue_level', 'new_user_class_level',
#     'hour', 'weekday', 'price',
#     'ad_ctr_cv', 'user_ctr_cv',
#     'brand_te', 'cate_id_te', 'pid_te', 'cms_group_id_te',
#     'gender_cate', 'age_cate', 'brand_cate',
#     'time_since_last_click', 'user_count_1H', 'user_ctr_1H',
#     'user_count_1D', 'user_ctr_1D', 'user_count_3D', 'user_ctr_3D',
#     'affinity_score', 'total_interactions', 'interest_recency' # lgbm features
# ]


# LABEL = {
#     "final_gender_code":     "Gender",
#     "age_level":             "Age bracket",
#     "occupation":            "Student",
#     "shopping_level":        "Shopping depth",
#     "cms_segid":             "Micro-segment ID",
#     "cms_group_id":           "Micro-segment group",  
#     "pvalue_level":          "Consumption level",
#     "new_user_class_level":  "City tier",
#     "hour":                  "Hour of day",
#     "weekday":               "Day of week",
#     "price":                 "Log price",
#     "ad_ctr_cv":             "Ad's historical CTR (cross-val)",
#     "user_ctr_cv":           "User's historical CTR (cross-val)",
#     "brand_te":             "Brand CTR (target-enc.)",
#     "cate_id_te":           "Category CTR (target-enc.)",
#     "pid_te":                "Placement CTR (target-enc.)",
#     "cms_group_id_te":      "Micro-segment group CTR (target-enc.)",    
#     "gender_cate":           "Gender × Category CTR",
#     "age_cate":              "Age × Category CTR",
#     "brand_cate":            "Brand × Category CTR",
#     "time_since_last_click": "Seconds since last click",
#     "user_count_1H":         "Impressions last hour",
#     "user_ctr_1H":           "CTR last hour",
#     "user_count_1D":         "Impressions last day",
#     "user_ctr_1D":           "CTR last day",
#     "user_count_3D":         "Impressions last 3 days",
#     "user_ctr_3D":           "CTR last 3 days",
 
#     # Affinity features
#     'affinity_score':     "User-category affinity score", 
#     'total_interactions': "Total interactions", 
#     'interest_recency':   "Interest recency" ,
# }

# # Pure-categorical features → selectbox with labels
# CATEGORICAL = {
#     "final_gender_code":    {-1: "Unknown", 1: "Male", 2: "Female"},
#     "age_level":            {-1: "Unknown", 1: "< 18", 2: "18–24", 3: "25–29",
#                              4: "30–34", 5: "35–39", 6: "40+"},
#     "pvalue_level":         {-1: "Unknown", 1: "Low", 2: "Mid", 3: "High"},
#     "shopping_level":       {-1: "Unknown", 1: "Shallow", 2: "Moderate", 3: "Deep"},
#     "occupation":           {-1: "Unknown", 0: "Not student", 1: "Student"},
#     "new_user_class_level": {-1: "Unknown", 1: "Tier 1", 2: "Tier 2",
#                              3: "Tier 3", 4: "Tier 4"},
#     "dayofweek":            {0: "Monday", 1: "Tuesday", 2: "Wednesday",
#                              3: "Thursday", 4: "Friday", 5: "Saturday",
#                              6: "Sunday"},
#     "is_weekend":           {0: "Weekday", 1: "Weekend"},
#     "price_bucket":         {0: "B0", 1: "B1", 2: "B2", 3: "B3", 4: "B4"},
# }

# # (min, max, step) — pulled from LightGBM feature_infos
# RANGES = {
#     "cms_segid":             (-1,       96,       1),
#     "customer":              (3,        255847,   1),
#     "price":                 (0.0099,   2.8377,   0.01),
#     "brand":                 (0.0,      1.0,      0.001),
#     "cate_id":               (0.0,      1.0,      0.001),
#     "pid":                   (0.0501,   0.0539,   0.0001),
#     "user_count_1H":         (0,        24,       1),
#     "user_ctr_1H":           (0.0,      1.0,      0.01),
#     "user_count_1D":         (0,        44,       1),
#     "user_ctr_1D":           (0.0,      1.0,      0.01),
#     "user_count_3D":         (0,        78,       1),
#     "user_ctr_3D":           (0.0,      1.0,      0.01),
#     "gender_cate":           (0.0,      1.0,      0.001),
#     "age_cate":              (0.0,      1.0,      0.001),
#     "hour_cate":             (0.0,      0.175,    0.001),
#     "user_cate_affinity":    (0.0,      1.0,      0.01),
#     "hour":                  (0,        23,       1),
#     "time_since_last_click": (0,        999999,   60),
# }

# GROUPS = {
#     "👤  User demographics": [
#         "final_gender_code", "age_level", "pvalue_level", "shopping_level",
#         "occupation", "new_user_class_level", "cms_segid",
#     ],
#     "🎯  Ad attributes": [
#         "cate_id", "brand", "customer", "price", "price_bucket", "pid",
#     ],
#     "📈  Behavior history": [
#         "user_count_1H", "user_ctr_1H", "user_count_1D", "user_ctr_1D",
#         "user_count_3D", "user_ctr_3D", "time_since_last_click",
#     ],
#     "🔗  Interactions (derived)": [
#         "user_cate_affinity", "gender_cate", "age_cate", "hour_cate",
#     ],
#     "🕒  Temporal context": [
#         "dayofweek", "is_weekend", "hour",
#     ],
# }

# # (phrase-when-SHAP-is-positive, phrase-when-SHAP-is-negative)
# NARRATIVE = {
#     "user_cate_affinity":    ("user shows strong past interest in this ad's category",
#                               "user has shown little past interest in this ad's category"),
#     "user_ctr_3D":           ("user has been clicking frequently in the last 3 days",
#                               "user has largely been ignoring ads in the last 3 days"),
#     "user_ctr_1D":           ("user is highly engaged today",
#                               "user has been unengaged today"),
#     "user_ctr_1H":           ("user is actively clicking right now",
#                               "user has been quiet in the last hour"),
#     "user_count_3D":         ("user has seen many ads recently (active browser)",
#                               "user has been seen few ads recently (cold start)"),
#     "user_count_1D":         ("user has been browsing heavily today",
#                               "user has barely browsed today"),
#     "user_count_1H":         ("user is active in-session",
#                               "user is not currently active"),
#     "cate_id":               ("ad's category has a historically high CTR",
#                               "ad's category has a historically low CTR"),
#     "brand":                 ("brand has a strong click track record",
#                               "brand has a weak click track record"),
#     "pid":                   ("placement slot performs well on average",
#                               "placement slot performs poorly on average"),
#     "gender_cate":           ("category resonates with the user's gender",
#                               "category rarely resonates with this gender"),
#     "age_cate":              ("category resonates with the user's age bracket",
#                               "category rarely resonates with this age bracket"),
#     "hour_cate":             ("category performs well at this time of day",
#                               "category performs poorly at this time of day"),
#     "price":                 ("price is attractive for this user",
#                               "price looks high given the context"),
#     "customer":              ("advertiser has a strong historical signal",
#                               "advertiser has a weak historical signal"),
#     "time_since_last_click": ("user clicked recently (high intent)",
#                               "user hasn't clicked in a long time"),
#     "age_level":             ("age bracket aligns with typical clickers",
#                               "age bracket rarely clicks this kind of ad"),
#     "final_gender_code":     ("gender matches typical clickers",
#                               "gender rarely clicks this kind of ad"),
#     "pvalue_level":          ("consumption tier is a strong buyer profile",
#                               "consumption tier rarely engages"),
#     "shopping_level":        ("deep shopper — naturally engaged",
#                               "shallow shopper — usually skims past"),
#     "new_user_class_level":  ("city tier is a hotspot for this ad",
#                               "city tier rarely responds"),
#     "hour":                  ("hour of day is prime-time for clicks",
#                               "hour of day is a dead zone for clicks"),
#     "is_weekend":            ("weekend effect favors the click",
#                               "weekend effect works against the click"),
#     "dayofweek":             ("day-of-week tilts positive",
#                               "day-of-week tilts negative"),
#     "occupation":            ("student status helps here",
#                               "student status hurts here"),
#     "cms_segid":             ("micro-segment leans toward clicks",
#                               "micro-segment leans away from clicks"),
#     "price_bucket":          ("price bucket favors clicks",
#                               "price bucket hurts clicks"),
# }

# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE + CSS
# # ═══════════════════════════════════════════════════════════════════════════════
# st.set_page_config(
#     page_title="CTR Sandbox · Module 4",
#     page_icon="🎯",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown(
#     """
#     <style>
#       .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
#       [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 600; }
#       [data-testid="stSidebar"] { background: #fafafa; }
#       [data-testid="stSidebar"] .stExpander { border: 1px solid #e5e7eb; border-radius: 8px; }
#       h1, h2, h3 { letter-spacing: -0.015em; }
#       .hero-title {
#           font-family: 'Georgia', serif;
#           font-size: 2.6rem; font-weight: 600;
#           margin: 0; padding: 0; color: #111;
#       }
#       .hero-kicker {
#           text-transform: uppercase; letter-spacing: 0.22em;
#           font-size: 0.75rem; color: #6366f1; font-weight: 700;
#           margin-bottom: 0.35rem;
#       }
#       .narrative {
#           background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 100%);
#           border-left: 4px solid #6366f1;
#           padding: 1.1rem 1.3rem;
#           border-radius: 10px;
#           line-height: 1.65;
#           font-size: 1.02rem;
#       }
#       .narrative ul { margin: 0.4rem 0 0 1.1rem; padding: 0; }
#       .narrative li { margin: 0.25rem 0; }
#       .chip-pos { color: #15803d; font-weight: 600; }
#       .chip-neg { color: #b91c1c; font-weight: 600; }
#       .tiny { color: #6b7280; font-size: 0.82rem; }
#       footer { visibility: hidden; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ═══════════════════════════════════════════════════════════════════════════════
# # HELPERS
# # ═══════════════════════════════════════════════════════════════════════════════
# def _py(val, feat):
#     """Cast a numpy/pandas scalar to a native Python value that matches the
#     widget type (int for integer sliders & categoricals, float otherwise),
#     clipped into the feature's valid range."""
#     if hasattr(val, "item"):
#         val = val.item()
#     if feat in CATEGORICAL:
#         options = CATEGORICAL[feat]
#         v = int(round(float(val)))
#         return v if v in options else next(iter(options))
#     step = RANGES.get(feat, (0, 0, 1))[2]
#     lo, hi, _ = RANGES[feat]
#     if isinstance(step, int):
#         return max(int(lo), min(int(hi), int(round(float(val)))))
#     return float(max(lo, min(hi, float(val))))

# # ═══════════════════════════════════════════════════════════════════════════════
# # LOADERS (cached)
# # ═══════════════════════════════════════════════════════════════════════════════
# @st.cache_resource(show_spinner="Loading model…")
# def load_model():
#     """Prefer joblib; fall back to LightGBM .txt."""
#     try:
#         return joblib.load(MODEL_JOBLIB), "joblib"
#     except Exception as err_joblib:
#         try:
#             import lightgbm as lgb
#             return (lgb.Booster(model_file=str(MODEL_TXT)),
#                     f"lightgbm .txt (joblib err: {type(err_joblib).__name__})")
#         except Exception as err_txt:
#             raise RuntimeError(
#                 f"Could not load model.\n  joblib: {err_joblib}\n  txt: {err_txt}"
#             )

# @st.cache_data(show_spinner="Loading background sample…")
# def load_background() -> pd.DataFrame:
#     if BG_PARQUET.exists():
#         df = pd.read_parquet(BG_PARQUET)
#     elif BG_CSV.exists():
#         df = pd.read_csv(BG_CSV)
#     else:
#         raise FileNotFoundError(
#             f"Need {BG_PARQUET.name} or {BG_CSV.name} next to app.py"
#         )
#     return df[FEATURE_ORDER].copy()

# @st.cache_resource(show_spinner="Building SHAP explainer…")
# def build_explainer(_model):
#     return shap.TreeExplainer(_model)

# @st.cache_data(show_spinner="Computing global importance…")
# def global_importance(_explainer, bg: pd.DataFrame, n_sample: int = 300) -> pd.DataFrame:
#     sample = bg.head(n_sample)
#     exp = _explainer(sample)
#     vals = np.asarray(exp.values)
#     if vals.ndim == 3:
#         vals = vals[:, :, -1]
#     mean_abs = np.abs(vals).mean(axis=0)
#     return (
#         pd.DataFrame({
#             "feature":       FEATURE_ORDER,
#             "label":         [LABEL[f] for f in FEATURE_ORDER],
#             "mean_abs_shap": mean_abs,
#         })
#         .sort_values("mean_abs_shap", ascending=False)
#         .reset_index(drop=True)
#     )

# # ═══════════════════════════════════════════════════════════════════════════════
# # PREDICT + EXPLAIN
# # ═══════════════════════════════════════════════════════════════════════════════
# def predict_ctr(model, row: pd.DataFrame) -> float:
#     if hasattr(model, "predict_proba"):
#         return float(model.predict_proba(row)[:, 1][0])
#     return float(model.predict(row)[0])   # LightGBM Booster w/ binary objective

# def explain_row(explainer, row: pd.DataFrame):
#     """Return (shap_values shape (n_feat,), base_value) in log-odds space."""
#     exp = explainer(row)
#     vals = np.asarray(exp.values)
#     base = np.asarray(exp.base_values)
#     if vals.ndim == 3:                     # (1, n_feat, n_class)
#         sv = vals[0, :, -1].astype(float)
#         bv = float(base[0, -1]) if base.ndim >= 2 else float(base[-1])
#     elif vals.ndim == 2:                   # (1, n_feat)
#         sv = vals[0].astype(float)
#         bv = float(base[0]) if base.ndim >= 1 else float(base)
#     else:                                  # (n_feat,)
#         sv = vals.astype(float)
#         bv = float(base)
#     return sv, bv

# def generate_narrative(shap_vals, values, top_k=3, threshold=0.01):
#     pairs = list(zip(FEATURE_ORDER, shap_vals))
#     pos = sorted([p for p in pairs if p[1] >  threshold], key=lambda x: -x[1])[:top_k]
#     neg = sorted([p for p in pairs if p[1] < -threshold], key=lambda x:  x[1])[:top_k]

#     def phrase(feat, sv):
#         if feat not in NARRATIVE:
#             return f"{LABEL.get(feat, feat)} = {values[feat]}"
#         p_pos, p_neg = NARRATIVE[feat]
#         return p_pos if sv > 0 else p_neg

#     return {
#         "positive": [(f, sv, phrase(f, sv)) for f, sv in pos],
#         "negative": [(f, sv, phrase(f, sv)) for f, sv in neg],
#     }

# # ═══════════════════════════════════════════════════════════════════════════════
# # PLOTS
# # ═══════════════════════════════════════════════════════════════════════════════
# def gauge(prob: float) -> go.Figure:
#     color = "#94a3b8" if prob < 0.02 else ("#f59e0b" if prob < 0.08 else "#22c55e")
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=prob * 100,
#         number={"suffix": "%", "font": {"size": 46, "family": "Georgia"}},
#         gauge={
#             "axis":   {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
#             "bar":    {"color": color, "thickness": 0.28},
#             "bgcolor": "white",
#             "borderwidth": 0,
#             "steps": [
#                 {"range": [0,   2], "color": "#f8fafc"},
#                 {"range": [2,   8], "color": "#fef3c7"},
#                 {"range": [8, 100], "color": "#dcfce7"},
#             ],
#         },
#         domain={"x": [0, 1], "y": [0, 1]},
#     ))
#     fig.update_layout(
#         margin=dict(l=10, r=10, t=10, b=10),
#         height=230,
#         paper_bgcolor="rgba(0,0,0,0)",
#     )
#     return fig

# def waterfall_fig(shap_vals, base_value, feat_vals, max_display=12):
#     explanation = shap.Explanation(
#         values=shap_vals,
#         base_values=base_value,
#         data=feat_vals,
#         feature_names=[LABEL[f] for f in FEATURE_ORDER],
#     )
#     plt.figure(figsize=(8, 5.5))
#     shap.plots.waterfall(explanation, max_display=max_display, show=False)
#     plt.tight_layout()
#     return plt.gcf()

# # ═══════════════════════════════════════════════════════════════════════════════
# # BOOT + SESSION-STATE INIT
# # ═══════════════════════════════════════════════════════════════════════════════
# model, model_source = load_model()
# bg = load_background()
# explainer = build_explainer(model)
# gimp = global_importance(explainer, bg)

# # Baseline = column-wise median of background sample
# BASELINE = {f: _py(bg[f].median(), f) for f in FEATURE_ORDER}

# for f in FEATURE_ORDER:
#     if f not in st.session_state:
#         st.session_state[f] = BASELINE[f]

# # ═══════════════════════════════════════════════════════════════════════════════
# # SIDEBAR — inputs
# # ═══════════════════════════════════════════════════════════════════════════════
# st.sidebar.markdown("### 🎛  Scenario controls")
# st.sidebar.caption("Adjust any feature — prediction & SHAP update live.")

# c1, c2 = st.sidebar.columns(2)
# if c1.button("🎲  Random user", use_container_width=True):
#     row = bg.sample(1).iloc[0]
#     for f in FEATURE_ORDER:
#         st.session_state[f] = _py(row[f], f)
#     st.rerun()
# if c2.button("↺  Reset to median", use_container_width=True):
#     for f in FEATURE_ORDER:
#         st.session_state[f] = BASELINE[f]
#     st.rerun()

# st.sidebar.markdown("---")

# def render_widget(feat: str):
#     label = LABEL[feat]
#     if feat in CATEGORICAL:
#         opts = list(CATEGORICAL[feat].keys())
#         st.selectbox(
#             label, opts,
#             index=opts.index(st.session_state[feat]),
#             format_func=lambda k: CATEGORICAL[feat][k],
#             key=feat,
#         )
#         return
#     lo, hi, step = RANGES[feat]
#     if isinstance(step, int):
#         st.slider(label, int(lo), int(hi), step=int(step), key=feat)
#     else:
#         fmt = "%.4f" if step < 0.01 else "%.3f"
#         st.slider(label, float(lo), float(hi), step=float(step),
#                   format=fmt, key=feat)

# for group_title, feats in GROUPS.items():
#     with st.sidebar.expander(group_title, expanded=group_title.startswith("👤")):
#         for f in feats:
#             render_widget(f)

# st.sidebar.markdown("---")
# st.sidebar.caption(f"Model source · `{model_source}`")

# # ═══════════════════════════════════════════════════════════════════════════════
# # BUILD INPUT ROW + PREDICT + SHAP
# # ═══════════════════════════════════════════════════════════════════════════════
# current_vals = {f: st.session_state[f] for f in FEATURE_ORDER}
# X_row  = pd.DataFrame([[current_vals[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
# X_base = pd.DataFrame([[BASELINE[f]     for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

# p_ctr  = predict_ctr(model, X_row)
# p_base = predict_ctr(model, X_base)
# shap_vals, base_lo = explain_row(explainer, X_row)

# narrative = generate_narrative(shap_vals, current_vals)

# # ═══════════════════════════════════════════════════════════════════════════════
# # MAIN — HERO
# # ═══════════════════════════════════════════════════════════════════════════════
# st.markdown('<div class="hero-kicker">Module 4 · Tab 2 · Prediction Sandbox</div>',
#             unsafe_allow_html=True)
# st.markdown('<h1 class="hero-title">Why did the model predict this CTR?</h1>',
#             unsafe_allow_html=True)
# st.markdown(
#     '<p class="tiny">A live, stakeholder-facing explanation of one specific '
#     'prediction — what the model thinks and which features got it there.</p>',
#     unsafe_allow_html=True,
# )
# st.write("")

# hero_l, hero_r = st.columns([0.42, 0.58], gap="large")

# with hero_l:
#     st.plotly_chart(gauge(p_ctr), use_container_width=True,
#                     config={"displayModeBar": False})
#     m1, m2, m3 = st.columns(3)
#     delta_pp = (p_ctr - p_base) * 100
#     m1.metric("Predicted CTR", f"{p_ctr*100:.2f}%",
#               f"{delta_pp:+.2f} pp vs baseline")
#     m2.metric("Baseline CTR", f"{p_base*100:.2f}%",
#               help="Median-user, median-ad from the background sample")
#     m3.metric("Σ SHAP", f"{shap_vals.sum():+.3f}",
#               help="Total log-odds shift from the model's baseline")

# with hero_r:
#     pos, neg = narrative["positive"], narrative["negative"]
#     parts = ['<div class="narrative">']
#     parts.append(
#         f"<b>The model predicts a {p_ctr*100:.1f}% click probability</b> "
#         f"(versus a {p_base*100:.1f}% baseline). Top drivers:"
#     )
#     parts.append("<ul>")
#     for f, sv, ph in pos:
#         parts.append(
#             f'<li><span class="chip-pos">↑ pushes up</span> — {ph} '
#             f'<span class="tiny">({sv:+.3f} log-odds)</span></li>'
#         )
#     for f, sv, ph in neg:
#         parts.append(
#             f'<li><span class="chip-neg">↓ pulls down</span> — {ph} '
#             f'<span class="tiny">({sv:+.3f} log-odds)</span></li>'
#         )
#     if not pos and not neg:
#         parts.append(
#             '<li>No single feature is swinging the prediction strongly — '
#             'this profile looks typical.</li>'
#         )
#     parts.append("</ul></div>")
#     st.markdown("\n".join(parts), unsafe_allow_html=True)

# st.write("")

# # ═══════════════════════════════════════════════════════════════════════════════
# # MAIN — DETAIL TABS
# # ═══════════════════════════════════════════════════════════════════════════════
# tab_local, tab_global, tab_data = st.tabs(
#     ["🔍  This prediction (local)", "🌐  Global importance", "📋  Raw input"]
# )

# with tab_local:
#     left, right = st.columns([0.58, 0.42], gap="large")
#     with left:
#         st.markdown("#### SHAP waterfall")
#         st.caption(
#             "Each bar is a feature's push — in log-odds — from the model's "
#             "baseline toward this prediction."
#         )
#         max_disp = st.slider("Show top N features", 5, 20, 12, 1,
#                              key="wf_max_display")
#         fig = waterfall_fig(shap_vals, base_lo,
#                             X_row.iloc[0].values, max_display=max_disp)
#         st.pyplot(fig, use_container_width=True, clear_figure=True)

#     with right:
#         st.markdown("#### Feature contributions")
#         df_contrib = (
#             pd.DataFrame({
#                 "Feature": [LABEL[f] for f in FEATURE_ORDER],
#                 "Value":   [current_vals[f] for f in FEATURE_ORDER],
#                 "SHAP":    shap_vals,
#             })
#             .sort_values("SHAP", key=np.abs, ascending=False)
#             .reset_index(drop=True)
#         )
#         st.dataframe(
#             df_contrib,
#             use_container_width=True,
#             height=480,
#             column_config={
#                 "Value": st.column_config.NumberColumn(format="%.4f"),
#                 "SHAP":  st.column_config.NumberColumn(
#                     "SHAP Δ (log-odds)", format="%+.3f"
#                 ),
#             },
#             hide_index=True,
#         )

# with tab_global:
#     st.markdown("#### Mean |SHAP| across the background sample")
#     st.caption(
#         "Which features matter most to the model overall? Cross-check this "
#         "against the Module 2 EDA — do the model's top drivers align with "
#         "the patterns you found in the data?"
#     )
#     top_n = st.slider("Top N", 5, 27, 15, 1, key="g_top_n")
#     g_view = gimp.head(top_n).iloc[::-1]
#     fig_g = go.Figure(go.Bar(
#         x=g_view["mean_abs_shap"],
#         y=g_view["label"],
#         orientation="h",
#         marker=dict(color="#6366f1"),
#         hovertemplate="<b>%{y}</b><br>mean |SHAP| = %{x:.4f}<extra></extra>",
#     ))
#     fig_g.update_layout(
#         height=40 + 24 * top_n,
#         margin=dict(l=10, r=10, t=10, b=30),
#         xaxis_title="mean |SHAP| (log-odds)",
#         yaxis_title="",
#         plot_bgcolor="white",
#         paper_bgcolor="rgba(0,0,0,0)",
#     )
#     st.plotly_chart(fig_g, use_container_width=True,
#                     config={"displayModeBar": False})

# with tab_data:
#     st.markdown("#### Current scenario — exact model input row")
#     display_df = X_row.T.rename(columns={0: "value"})
#     display_df["label"] = [LABEL[f] for f in FEATURE_ORDER]
#     st.dataframe(
#         display_df[["label", "value"]],
#         use_container_width=True,
#         height=520,
#     )
