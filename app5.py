import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
from xgboost import XGBClassifier

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #141824 100%);
        border-right: 1px solid #2d3348;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #e84545;
        font-size: 1.1rem;
        border-bottom: 1px solid #2d3348;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    .predict-high {
        background: linear-gradient(135deg, #3d1a1a 0%, #2d1515 100%);
        border: 2px solid #e84545;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
    }
    .predict-low {
        background: linear-gradient(135deg, #0f2d1a 0%, #0d2415 100%);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
    }
    .predict-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; }
    .predict-subtitle { color: #a0aec0; font-size: 0.95rem; margin-top: 8px; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c8d0e0;
        margin: 20px 0 12px 0;
        padding-left: 10px;
        border-left: 3px solid #e84545;
    }

    .welcome-box {
        background: linear-gradient(135deg, #1e2235 0%, #252b40 100%);
        border: 1px solid #2d3348;
        border-radius: 16px;
        padding: 60px 40px;
        text-align: center;
        margin: 40px 0;
    }

    hr { border-color: #2d3348 !important; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("best_xgboost_model.json")
    return model

@st.cache_resource
def load_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl"), True
    return None, False

model = load_model()
scaler, scaler_loaded = load_scaler()

# ─── Scaler fallback ────────────────────────────────────────────────────────────
SCALER_MEAN = {"BMI": 28.33, "PhysicalHealth": 3.37, "MentalHealth": 3.90,
               "SleepTime": 7.10, "AgeCategory": 55.80, "GenHealth": 2.52}
SCALER_STD  = {"BMI": 6.36,  "PhysicalHealth": 7.67, "MentalHealth": 7.95,
               "SleepTime": 1.43, "AgeCategory": 15.45, "GenHealth": 1.07}
NUM_COLS = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']

FEATURE_ORDER = [
    'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
    'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth',
    'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer',
    'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White',
]

# Fallback only — used if the model file can't be parsed for any reason.
_FALLBACK_IMPORTANCE = {
    'AgeCategory': 0.142, 'BMI': 0.118, 'GenHealth': 0.107, 'PhysicalHealth': 0.098,
    'Stroke': 0.084, 'DiffWalking': 0.071, 'Diabetic': 0.065, 'MentalHealth': 0.058,
    'SleepTime': 0.045, 'Smoking': 0.038, 'KidneyDisease': 0.034, 'SkinCancer': 0.028,
    'Sex': 0.025, 'PhysicalActivity': 0.022, 'AlcoholDrinking': 0.018,
}

@st.cache_resource
def compute_feature_importance():
    """Compute real gain-based feature importances from the trained XGBoost model.

    Read straight from the saved model so the chart always reflects the
    model that is actually shipped, rather than hardcoded placeholder numbers.
    """
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")  # {feature: total_gain}
        if not score:
            return dict(_FALLBACK_IMPORTANCE)
        total = sum(score.values())
        imp = {f: v / total for f, v in score.items()}
        # Ensure every modelled feature has an entry (features never used as a
        # split get 0), then sort descending by importance.
        for f in FEATURE_ORDER:
            imp.setdefault(f, 0.0)
        return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
    except Exception:
        return dict(_FALLBACK_IMPORTANCE)

TOP_FEATURES_IMPORTANCE = compute_feature_importance()

# ─── Preprocessing ───────────────────────────────────────────────────────────────
def preprocess(inputs):
    row = {feat: inputs.get(feat, 0) for feat in FEATURE_ORDER}
    df_row = pd.DataFrame([row])[FEATURE_ORDER]
    if scaler_loaded and scaler is not None:
        df_row[NUM_COLS] = scaler.transform(df_row[NUM_COLS])
    else:
        for col in NUM_COLS:
            df_row[col] = (df_row[col] - SCALER_MEAN[col]) / SCALER_STD[col]
    return df_row

# ─── Session State ───────────────────────────────────────────────────────────────
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "prob" not in st.session_state:
    st.session_state.prob = None
if "pred" not in st.session_state:
    st.session_state.pred = None
if "snap" not in st.session_state:
    st.session_state.snap = None

# Seed slider defaults once on first load (sliders no longer pass positional
# defaults, so without this they would start at their minimum).
for _k, _v in {"bmi": 27.0, "sleep_time": 7, "physical_health": 0, "mental_health": 0}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Process a pending reset BEFORE any widget is created. Explicitly assign each
# widget key back to its default value — this is honored reliably across
# Streamlit versions (popping keys can be re-restored by the widget on rerun).
if st.session_state.get("_do_reset", False):
    _defaults = {
        "age_idx": None, "sex": None, "race": None, "physical_activity": None,
        "smoking": None, "alcohol": None, "gen_health": None, "diff_walking": None,
        "stroke": None, "diabetic": None, "asthma": None, "kidney_disease": None,
        "skin_cancer": None, "bmi": 27.0, "sleep_time": 7,
        "physical_health": 0, "mental_health": 0,
    }
    for k, v in _defaults.items():
        st.session_state[k] = v
    st.session_state.predicted = False
    st.session_state.prob = None
    st.session_state.pred = None
    st.session_state.snap = None
    st.session_state._do_reset = False

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Patient Profile")

    st.markdown("**Demographics**")
    age_options = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    age_labels  = ["18–24","25–29","30–34","35–39","40–44","45–49",
                   "50–54","55–59","60–64","65–69","70–74","75–79","80+"]
    age_idx = st.selectbox("Age Group", [None] + list(range(len(age_labels))),
                           format_func=lambda i: "— Select —" if i is None else age_labels[i],
                           key="age_idx")
    age_val = age_options[age_idx] if age_idx is not None else None

    sex  = st.selectbox("Sex", [None, "Female", "Male"],
                        format_func=lambda x: "— Select —" if x is None else x, key="sex")
    sex_val = (1 if sex == "Male" else 0) if sex is not None else None

    race = st.selectbox("Race / Ethnicity",
        [None, "American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "Other", "White"],
        format_func=lambda x: "— Select —" if x is None else x, key="race")
    race_map = {"American Indian/Alaskan Native": "none", "Asian": "Race_Asian",
                "Black": "Race_Black", "Hispanic": "Race_Hispanic",
                "Other": "Race_Other", "White": "Race_White"}
    race_key = race_map[race] if race is not None else None

    st.markdown("---")
    st.markdown("**Body & Lifestyle**")
    bmi              = st.slider("BMI", 10.0, 60.0, step=0.5, key="bmi")
    sleep_time       = st.slider("Sleep Hours / Night", 1, 24, key="sleep_time")
    physical_activity= st.selectbox("Physically Active (past 30 days)", [None, "Yes", "No"],
                                    format_func=lambda x: "— Select —" if x is None else x,
                                    key="physical_activity")
    smoking          = st.selectbox("Smoking (100+ cigarettes lifetime)", [None, "No", "Yes"],
                                    format_func=lambda x: "— Select —" if x is None else x,
                                    key="smoking")
    alcohol          = st.selectbox("Heavy Alcohol Drinking", [None, "No", "Yes"],
                                    format_func=lambda x: "— Select —" if x is None else x,
                                    key="alcohol")

    st.markdown("---")
    st.markdown("**Health Conditions**")
    gen_health_opts = ["Poor", "Fair", "Good", "Very good", "Excellent"]
    gen_health      = st.selectbox("General Health", [None] + gen_health_opts,
                                   format_func=lambda x: "— Select —" if x is None else x,
                                   key="gen_health")
    gen_health_val  = gen_health_opts.index(gen_health) if gen_health is not None else None

    physical_health = st.slider("Physically Unhealthy Days (past 30 days)", 0, 30, key="physical_health")
    mental_health   = st.slider("Mentally Unhealthy Days (past 30 days)", 0, 30, key="mental_health")
    diff_walking    = st.selectbox("Difficulty Walking / Climbing Stairs", [None, "No", "Yes"],
                                   format_func=lambda x: "— Select —" if x is None else x, key="diff_walking")
    stroke          = st.selectbox("Ever Had a Stroke?", [None, "No", "Yes"],
                                   format_func=lambda x: "— Select —" if x is None else x, key="stroke")
    diabetic        = st.selectbox("Diabetic?", [None, "No", "Yes"],
                                   format_func=lambda x: "— Select —" if x is None else x, key="diabetic")
    asthma          = st.selectbox("Asthma", [None, "No", "Yes"],
                                   format_func=lambda x: "— Select —" if x is None else x, key="asthma")
    kidney_disease  = st.selectbox("Kidney Disease", [None, "No", "Yes"],
                                   format_func=lambda x: "— Select —" if x is None else x, key="kidney_disease")
    skin_cancer     = st.selectbox("Skin Cancer", [None, "No", "Yes"],
                                   format_func=lambda x: "— Select —" if x is None else x, key="skin_cancer")

    st.markdown("---")
    b1, b2 = st.columns(2)
    predict_btn = b1.button("🔍  Predict", use_container_width=True, type="primary")
    reset_btn   = b2.button("♻️  Reset", use_container_width=True)

# ─── Reset: request a clear, handled at the top of the next run ───────────────────
if reset_btn:
    st.session_state._do_reset = True
    st.rerun()

# ─── Build Input Dict ─────────────────────────────────────────────────────────────
def yn(v): return 1 if v == "Yes" else 0

# The dropdowns that must be chosen before a prediction can run.
_required = [age_val, sex_val, race_key, physical_activity, smoking, alcohol,
             gen_health_val, diff_walking, stroke, diabetic, asthma,
             kidney_disease, skin_cancer]
all_filled = all(v is not None for v in _required)

# ─── Run Prediction on button click ─────────────────────────────────────────────
if predict_btn:
    if not all_filled:
        st.sidebar.warning("Please fill in every field before predicting.")
    else:
        inputs = {
            "BMI": bmi, "Smoking": yn(smoking), "AlcoholDrinking": yn(alcohol),
            "Stroke": yn(stroke), "PhysicalHealth": float(physical_health),
            "MentalHealth": float(mental_health), "DiffWalking": yn(diff_walking),
            "Sex": sex_val, "AgeCategory": float(age_val), "Diabetic": yn(diabetic),
            "PhysicalActivity": yn(physical_activity), "GenHealth": float(gen_health_val),
            "SleepTime": float(sleep_time), "Asthma": yn(asthma),
            "KidneyDisease": yn(kidney_disease), "SkinCancer": yn(skin_cancer),
            "Race_Asian": 0, "Race_Black": 0, "Race_Hispanic": 0, "Race_Other": 0, "Race_White": 0,
        }
        if race_key != "none":
            inputs[race_key] = 1
        X = preprocess(inputs)
        st.session_state.prob = float(model.predict_proba(X)[0][1])
        st.session_state.pred = int(model.predict(X)[0])
        st.session_state.snap = inputs.copy()
        st.session_state.predicted = True

# ─── Header ──────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<p style='font-size:3rem; margin-top:8px;'>❤️</p>", unsafe_allow_html=True)
with col_title:
    st.markdown("## Heart Disease Prediction Dashboard")
    st.caption("WQD7001 · Principles of Data Science · Group Project")

st.markdown("---")

# ─── Welcome Screen (before prediction) ─────────────────────────────────────────
if not st.session_state.predicted:
    st.markdown("""
    <div class="welcome-box">
        <p style="font-size:3rem; margin-bottom:16px;">🫀</p>
        <p style="font-size:1.4rem; font-weight:700; color:#c8d0e0; margin-bottom:10px;">
            Heart Disease Risk Assessment
        </p>
        <p style="color:#8892a4; font-size:1rem; max-width:500px; margin:0 auto 24px;">
            Fill in the patient details in the sidebar on the left,
            then click <strong style="color:#e84545">Predict Now</strong> to see the results.
        </p>
        <p style="color:#4a5568; font-size:0.85rem;">
            Powered by XGBoost
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Results (after prediction) ─────────────────────────────────────────────────
else:
    prob = st.session_state.prob
    pred = st.session_state.pred
    snap = st.session_state.snap

    # ── Row 1: Prediction result + Gauge ─────────────────────────────────────────
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<p class="section-header">🩺 Prediction Result</p>', unsafe_allow_html=True)
        if pred == 1:
            st.markdown(f"""
            <div class="predict-high">
                <div class="predict-title" style="color:#e84545">⚠️ Heart Disease Detected</div>
                <div class="predict-subtitle">
                    The model predicts a <b style="color:#e84545">{prob:.1%}</b> probability of heart disease.<br>
                    Please consult a medical professional for a full diagnosis.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="predict-low">
                <div class="predict-title" style="color:#22c55e">✅ No Heart Disease Detected</div>
                <div class="predict-subtitle">
                    The model predicts a <b style="color:#22c55e">{prob:.1%}</b> probability of heart disease.<br>
                    Maintain a healthy lifestyle and have regular check-ups.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk Flags
        st.markdown('<p class="section-header">🚩 Patient Risk Flags</p>', unsafe_allow_html=True)
        flags = {
            "Stroke History":              snap["Stroke"] == 1,
            "Diabetic":                    snap["Diabetic"] == 1,
            "Difficulty Walking":          snap["DiffWalking"] == 1,
            "Smoker":                      snap["Smoking"] == 1,
            "Kidney Disease":              snap["KidneyDisease"] == 1,
            "BMI Overweight (≥ 25)":       snap["BMI"] >= 25,
            "Poor / Fair General Health":  snap["GenHealth"] <= 1,
            "Physically Inactive":         snap["PhysicalActivity"] == 0,
            "High Mental Burden (≥15 days)": snap["MentalHealth"] >= 15,
        }
        cols_f = st.columns(2)
        for i, (name, present) in enumerate(flags.items()):
            icon  = "🔴" if present else "🟢"
            color = "#fca5a5" if present else "#6ee7b7"
            cols_f[i % 2].markdown(
                f"<div style='padding:4px 0;font-size:0.86rem;color:{color}'>{icon} {name}</div>",
                unsafe_allow_html=True)

    with right:
        # Gauge
        st.markdown('<p class="section-header">📊 Heart Disease Probability</p>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568",
                         "tickfont": {"color": "#9aa3b2"}},
                "bar": {"color": "#e84545" if prob >= 0.5 else ("#f59e0b" if prob >= 0.35 else "#22c55e")},
                "bgcolor": "#1e2235", "borderwidth": 0,
                "steps": [
                    {"range": [0,  35],  "color": "#0f2d1a"},
                    {"range": [35, 60],  "color": "#2d2a0f"},
                    {"range": [60, 100], "color": "#2d0f0f"},
                ],
                "threshold": {"line": {"color": "white", "width": 2},
                              "thickness": 0.75, "value": prob * 100},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=20, r=20, t=20, b=10), font=dict(color="white"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        # Feature Importance
        st.markdown('<p class="section-header">📈 Top Feature Importances</p>', unsafe_allow_html=True)
        # Take the top 15 by importance, then sort ascending so the largest
        # bar sits at the top of the horizontal chart.
        _top15 = dict(sorted(TOP_FEATURES_IMPORTANCE.items(),
                             key=lambda kv: kv[1], reverse=True)[:15])
        feat_df = pd.DataFrame({
            "Feature": list(_top15.keys()),
            "Importance": list(_top15.values()),
        }).sort_values("Importance")

        high_risk_feats = ["Stroke", "AgeCategory", "Diabetic", "DiffWalking", "KidneyDisease"]
        colors_bar = ["#e84545" if f in high_risk_feats else "#60a5fa" for f in feat_df["Feature"]]

        fig_fi = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.3f}" for v in feat_df["Importance"]],
            textposition="outside", textfont=dict(color="#9aa3b2", size=10),
            cliponaxis=False,
        ))
        _fi_max = float(feat_df["Importance"].max())
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=340, margin=dict(l=10, r=90, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                       range=[0, _fi_max * 1.30]),
            yaxis=dict(tickfont=dict(color="#c8d0e0", size=11)), bargap=0.25,
        )
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── Row 2: Three charts ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="large")

    # Chart 1 — Patient Variable Bar Chart vs healthy range
    with c1:
        st.markdown('<p class="section-header">📋 Patient Profile vs Healthy Range</p>', unsafe_allow_html=True)

        # Reference healthy ranges (midpoint of healthy range)
        ref = {"BMI": 22.0, "PhysicalHealth": 0.0, "MentalHealth": 0.0, "SleepTime": 8.0}
        patient_vals = {
            "BMI":            snap["BMI"],
            "Physical\nUnhealthy Days": snap["PhysicalHealth"],
            "Mental\nUnhealthy Days":   snap["MentalHealth"],
            "Sleep\nHours":             snap["SleepTime"],
        }
        ref_vals = {
            "BMI":            22.0,
            "Physical\nUnhealthy Days": 0.0,
            "Mental\nUnhealthy Days":   0.0,
            "Sleep\nHours":             8.0,
        }
        labels = list(patient_vals.keys())
        p_vals = list(patient_vals.values())
        r_vals = list(ref_vals.values())

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Patient", x=labels, y=p_vals,
            marker_color="#e84545", text=[f"{v:.1f}" for v in p_vals],
            textposition="outside", textfont=dict(color="#c8d0e0", size=11),
        ))
        fig_bar.add_trace(go.Bar(
            name="Healthy Reference", x=labels, y=r_vals,
            marker_color="#22c55e", text=[f"{v:.1f}" for v in r_vals],
            textposition="outside", textfont=dict(color="#c8d0e0", size=11),
        ))
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color="#9aa3b2", size=11), bgcolor="rgba(0,0,0,0)",
                        orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(tickfont=dict(color="#c8d0e0", size=11), gridcolor="#2d3348"),
            yaxis=dict(tickfont=dict(color="#9aa3b2", size=10), gridcolor="#2d3348"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # Chart 2 — Risk Factor Pie Chart
    with c2:
        st.markdown('<p class="section-header">🥧 Risk Factor Breakdown</p>', unsafe_allow_html=True)

        risk_factors = {
            "Stroke":          snap["Stroke"],
            "Diabetic":        snap["Diabetic"],
            "Diff Walking":    snap["DiffWalking"],
            "Smoking":         snap["Smoking"],
            "Kidney Disease":  snap["KidneyDisease"],
            "Asthma":          snap["Asthma"],
            "Skin Cancer":     snap["SkinCancer"],
            "Heavy Alcohol":   snap["AlcoholDrinking"],
            "Inactive":        1 - snap["PhysicalActivity"],
        }
        present  = sum(risk_factors.values())
        absent   = len(risk_factors) - present

        fig_pie = go.Figure(go.Pie(
            labels=["Risk Factors Present", "Risk Factors Absent"],
            values=[present, absent],
            hole=0.55,
            marker=dict(colors=["#e84545", "#22c55e"],
                        line=dict(color="#0f1117", width=2)),
            textfont=dict(color="white", size=12),
            hovertemplate="%{label}: %{value}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{present}</b><br><span style='font-size:10px'>of 9</span>",
            x=0.5, y=0.5, font=dict(size=18, color="white"),
            showarrow=False, align="center",
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(color="#9aa3b2", size=11), bgcolor="rgba(0,0,0,0)",
                        orientation="h", yanchor="bottom", y=-0.15),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # Chart 3 — Feature Contribution for this patient
    with c3:
        st.markdown('<p class="section-header">⚡ This Patient\'s Risk Contributions</p>', unsafe_allow_html=True)

        # Weight each feature importance by how abnormal the patient's value is
        contrib = {
            "Age":             TOP_FEATURES_IMPORTANCE["AgeCategory"]    * (snap["AgeCategory"] / 80),
            "BMI":             TOP_FEATURES_IMPORTANCE["BMI"]             * min(snap["BMI"] / 40, 1),
            "Gen Health":      TOP_FEATURES_IMPORTANCE["GenHealth"]       * ((4 - snap["GenHealth"]) / 4),
            "Physical Health": TOP_FEATURES_IMPORTANCE["PhysicalHealth"]  * (snap["PhysicalHealth"] / 30),
            "Stroke":          TOP_FEATURES_IMPORTANCE["Stroke"]          * snap["Stroke"],
            "Diff Walking":    TOP_FEATURES_IMPORTANCE["DiffWalking"]     * snap["DiffWalking"],
            "Diabetic":        TOP_FEATURES_IMPORTANCE["Diabetic"]        * snap["Diabetic"],
            "Mental Health":   TOP_FEATURES_IMPORTANCE["MentalHealth"]    * (snap["MentalHealth"] / 30),
            "Sleep":           TOP_FEATURES_IMPORTANCE["SleepTime"]       * abs(snap["SleepTime"] - 8) / 8,
            "Smoking":         TOP_FEATURES_IMPORTANCE["Smoking"]         * snap["Smoking"],
        }
        contrib_df = pd.DataFrame({
            "Factor": list(contrib.keys()),
            "Score":  list(contrib.values()),
        })
        # Express each factor as a percentage of this patient's total risk drivers.
        _total = contrib_df["Score"].sum()
        contrib_df["Pct"] = (contrib_df["Score"] / _total * 100) if _total > 0 else 0.0
        contrib_df = contrib_df.sort_values("Pct", ascending=True)

        bar_colors = ["#e84545" if p > 10 else "#60a5fa" for p in contrib_df["Pct"]]

        _pct_max = float(contrib_df["Pct"].max()) if _total > 0 else 1.0
        fig_contrib = go.Figure(go.Bar(
            x=contrib_df["Pct"], y=contrib_df["Factor"], orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in contrib_df["Pct"]],
            textposition="outside", textfont=dict(color="#9aa3b2", size=10),
            cliponaxis=False,
        ))
        fig_contrib.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=80, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                       range=[0, _pct_max * 1.35]),
            yaxis=dict(tickfont=dict(color="#c8d0e0", size=11)),
            bargap=0.25,
        )
        st.plotly_chart(fig_contrib, use_container_width=True, config={"displayModeBar": False})

    # ── Disclaimer ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#4a5568; font-size:0.80rem; padding:10px">
        ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and does not constitute medical advice.
        Always consult a qualified healthcare professional for medical decisions.<br>
        WQD7001 · Principles of Data Science · Group Project · Heart Disease Prediction Using Machine Learning
    </div>""", unsafe_allow_html=True)
