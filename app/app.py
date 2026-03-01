"""
app.py
------
Streamlit dashboard for the Smart ETA & Delay Risk Predictor.

Run:
    streamlit run app.py

Requirements:
    pip install streamlit scikit-learn joblib numpy pandas
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart ETA & Delay Risk Predictor",
    page_icon="🛵",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background: #0d0d0d;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #ff4500 0%, #ff8c00 60%, #ffd700 100%);
    border-radius: 20px;
    padding: 36px 40px 28px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🛵";
    font-size: 120px;
    position: absolute;
    right: 20px;
    top: -10px;
    opacity: 0.18;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.1rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-weight: 300;
}

/* ── Input card ── */
.input-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.input-card h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #ff6b2b;
    margin: 0 0 20px;
}

/* ── Result cards ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
}
.result-card {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.result-card .label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
    font-weight: 600;
    margin-bottom: 8px;
}
.result-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}
.result-card .unit {
    font-size: 0.85rem;
    color: #888;
    margin-top: 4px;
}

/* ── Risk badge ── */
.risk-badge {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 24px;
}
.risk-badge .risk-label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 8px;
}
.risk-badge .risk-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.risk-low    { background: #0d2b1a; border: 1px solid #1a5c32; }
.risk-low    .risk-label { color: #4ade80; }
.risk-low    .risk-value { color: #4ade80; }
.risk-medium { background: #2b1f0d; border: 1px solid #7a4e0a; }
.risk-medium .risk-label { color: #fb923c; }
.risk-medium .risk-value { color: #fb923c; }
.risk-high   { background: #2b0d0d; border: 1px solid #7a0a0a; }
.risk-high   .risk-label { color: #f87171; }
.risk-high   .risk-value { color: #f87171; }

/* ── Progress bar ── */
.prob-bar-wrap {
    background: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
}
.prob-bar-wrap .label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
    font-weight: 600;
    margin-bottom: 12px;
}
.prob-track {
    background: #2e2e2e;
    border-radius: 999px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin-bottom: 8px;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
.prob-pct {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
}

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #ff4500, #ff8c00) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
div.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── Slider / select labels ── */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #aaa !important;
    letter-spacing: 0.3px !important;
}

/* ── Divider ── */
hr { border-color: #2e2e2e !important; }

/* ── Warning / info ── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loader (cached)
# ---------------------------------------------------------------------------

MODEL_PATHS = [
    "src/models/saved_model.pkl",
    "src/models/saved_model.joblib",
    "saved_model.pkl",
    "saved_model.joblib",
]

@st.cache_resource
def load_model():
    """Try common paths for the trained model artifact."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None


# ---------------------------------------------------------------------------
# Feature encoding helpers
# ---------------------------------------------------------------------------

# These must match the training feature order exactly.
FEATURE_COLS = [
    "prep_time_minutes",
    "distance_km",
    "traffic_index",
    "weather_score",
    "rider_experience_years",
    "is_peak_hour",
]

TRAFFIC_MAP = {"Low (1–2)": 2, "Medium (3)": 3, "High (4–5)": 5}
WEATHER_MAP = {"Clear": 0.05, "Light Rain": 0.35, "Heavy Rain / Storm": 0.80}


def is_peak(hour: int) -> int:
    return 1 if (12 <= hour <= 14 or 19 <= hour <= 22) else 0


def build_feature_df(
    prep_time: int,
    distance: float,
    traffic_label: str,
    weather_label: str,
    order_hour: int,
    rider_exp: float,
) -> pd.DataFrame:
    """Encode user inputs into the exact feature vector used during training."""
    return pd.DataFrame([{
        "prep_time_minutes":      float(prep_time),
        "distance_km":            float(distance),
        "traffic_index":          float(TRAFFIC_MAP[traffic_label]),
        "weather_score":          float(WEATHER_MAP[weather_label]),
        "rider_experience_years": float(rider_exp),
        "is_peak_hour":           float(is_peak(order_hour)),
    }])[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Risk classification helper
# ---------------------------------------------------------------------------

def classify_risk(eta_minutes: float, delay_prob: float | None) -> tuple[str, str]:
    """
    Returns (risk_level, css_class).
    Uses delay_prob if available, else falls back to ETA threshold.
    """
    score = delay_prob if delay_prob is not None else (1.0 if eta_minutes > 50 else 0.0)
    if score < 0.30:
        return "🟢 Low Risk",    "risk-low"
    elif score < 0.65:
        return "🟠 Medium Risk", "risk-medium"
    else:
        return "🔴 High Risk",   "risk-high"


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Delivery ETA & Delay Risk Predictor</h1>
  <p>Real-time ETA estimation and delay risk scoring powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

# ── Model status ─────────────────────────────────────────────────────────────
model, model_path = load_model()
if model is None:
    st.warning(
        "⚠️  No trained model found. Searched: " + ", ".join(MODEL_PATHS) +
        "\n\nTrain the model first with: `python src/models/model_training.py`"
    )
    st.stop()
else:
    st.success(f"✅  Model loaded from `{model_path}`", icon="🤖")

# ── Input card ───────────────────────────────────────────────────────────────
st.markdown('<div class="input-card"><h3>Order Parameters</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    distance    = st.slider("Distance (km)",          min_value=1.0,  max_value=12.0, value=6.0,  step=0.5)
    prep_time   = st.slider("Preparation Time (min)", min_value=10,   max_value=35,   value=20,   step=1)
    order_hour  = st.slider("Order Hour (0–23)",      min_value=0,    max_value=23,   value=13,   step=1)

with col2:
    traffic_label = st.selectbox("Traffic Level",  list(TRAFFIC_MAP.keys()), index=1)
    weather_label = st.selectbox("Weather",        list(WEATHER_MAP.keys()), index=0)
    rider_exp     = st.slider("Rider Experience (years)", min_value=0.5, max_value=5.0, value=2.5, step=0.5)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍  Predict ETA & Delay Risk")

# ── Results ──────────────────────────────────────────────────────────────────
if predict_clicked:
    X = build_feature_df(prep_time, distance, traffic_label, weather_label, order_hour, rider_exp)

    # ETA prediction
    eta_pred = float(model.predict(X)[0])
    eta_pred = max(5.0, round(eta_pred, 1))

    # Delay probability (if model supports predict_proba)
    delay_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        delay_prob = float(proba[1]) if len(proba) == 2 else None

    risk_label, risk_css = classify_risk(eta_pred, delay_prob)

    # ── ETA + delay prob row ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="result-grid">', unsafe_allow_html=True)

    # ETA card
    st.markdown(f"""
    <div class="result-card">
        <div class="label">Predicted ETA</div>
        <div class="value">{int(eta_pred)}</div>
        <div class="unit">minutes</div>
    </div>
    """, unsafe_allow_html=True)

    # Delay probability card
    if delay_prob is not None:
        dp_pct = round(delay_prob * 100, 1)
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Delay Probability</div>
            <div class="value">{dp_pct}<span style="font-size:1.2rem;color:#888">%</span></div>
            <div class="unit">chance of arriving late</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Derive a simple heuristic probability from ETA
        h_prob = min(100, max(0, int((eta_pred - 30) * 2)))
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated Delay Chance</div>
            <div class="value">{h_prob}<span style="font-size:1.2rem;color:#888">%</span></div>
            <div class="unit">heuristic estimate</div>
        </div>
        """, unsafe_allow_html=True)
        delay_prob = h_prob / 100.0

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Delay probability progress bar ───────────────────────────────────
    dp_display  = round(delay_prob * 100, 1)
    bar_color   = "#4ade80" if dp_display < 30 else ("#fb923c" if dp_display < 65 else "#f87171")
    st.markdown(f"""
    <div class="prob-bar-wrap">
        <div class="label">Delay Risk Meter</div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{dp_display}%; background:{bar_color};"></div>
        </div>
        <div class="prob-pct">{dp_display}% probability of delay</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk badge ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="risk-badge {risk_css}">
        <div class="risk-label">Overall Risk Level</div>
        <div class="risk-value">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Breakdown summary ─────────────────────────────────────────────────
    with st.expander("📊  Feature breakdown"):
        breakdown = {
            "Prep time contribution":     f"+{prep_time} min",
            "Distance contribution":      f"+{round(distance * 4, 1)} min  ({distance} km × 4)",
            "Traffic contribution":       f"+{TRAFFIC_MAP[traffic_label] * 3} min  (index {TRAFFIC_MAP[traffic_label]} × 3)",
            "Rider efficiency bonus":     f"−{round(rider_exp * 1.5, 1)} min  ({rider_exp} yrs × 1.5)",
            "Peak hour":                  "Yes ⚡" if is_peak(order_hour) else "No",
            "Weather":                    weather_label,
            "Final predicted ETA":        f"{eta_pred} min",
        }
        for k, v in breakdown.items():
            c1, c2 = st.columns([2, 1])
            c1.markdown(f"<span style='color:#888;font-size:0.88rem'>{k}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='color:#fff;font-size:0.88rem;font-weight:500'>{v}</span>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#444;font-size:0.78rem;font-family:DM Sans'>"
    "Smart ETA & Delay Risk Engine &nbsp;·&nbsp; Hackathon Build</p>",
    unsafe_allow_html=True
)