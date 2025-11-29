import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# ================================
# Load Model
# ================================
MODEL_PATH = "best_pipeline.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

pipe = joblib.load(MODEL_PATH)

# ================================
# Page Config + CSS (from output.py)
# ================================
st.set_page_config(page_title="Health Risk Predictor from Lifestyle & Medical History", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container { background: #0f1720; color: #e6eef8; }
    .stButton>button { background-color: #0ea5b7; color: white; }
    .card {
        background: #0b1220;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    .card h4 { margin: 0; color: #e6eef8; }
    .value { font-size: 22px; font-weight: 700; color: #60a5fa; margin-top: 6px; margin-bottom: 6px; }
    .meta { color:#9ca3af; font-size:13px }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# Title
# ================================
st.title("Health Risk Prediction System")
st.write("Predict risk percentages using lifestyle and clinical data.")

# ================================
# INPUT SECTION (from inputs.py)
# ================================
st.header("Provide Your Information")

colA, colB, colC = st.columns(3)
with colA:
    Age = st.number_input("Age", 1, 120, 30)
with colB:
    bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
with colC:
    systolic_bp = st.number_input("Systolic BP", 80.0, 200.0, 120.0)

with colA:
    diastolic_bp = st.number_input("Diastolic BP", 40.0, 140.0, 80.0)
with colB:
    hba1c = st.number_input("HbA1c (%)", 3.0, 12.0, 5.5)
with colC:
    glucose_fasting = st.number_input("Fasting Glucose", 50.0, 300.0, 95.0)

st.markdown("---")

# Demographics


with st.expander("Demographics"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
        smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "Former", "Current"], index=0)
    with col2:
        alcohol_consumption_per_week = st.slider("Alcohol (units/week)", 0, 30, 0)
        waist_to_hip_ratio = st.number_input("Waist-to-Hip Ratio", 0.6, 1.5, 0.9, step=0.01)


# Medical history
with st.expander("Medical History"):
    col1, col2, col3 = st.columns(3)
    with col1:
        family_history_diabetes = st.selectbox("Family History of Diabetes", ["Yes", "No"])
    with col2:
        hypertension_history = st.selectbox("Hypertension History", ["Yes", "No"])
    with col3:
        cardiovascular_history = st.selectbox("Cardiovascular History", ["Yes", "No"])

# Labs
with st.expander("Lab Tests & Vital Signs"):
    col1, col2 = st.columns(2)
    with col1:
        cholesterol_total = st.number_input("Total Cholesterol", 100.0, 400.0, 180.0)
        hdl_cholesterol = st.number_input("HDL Cholesterol", 20.0, 100.0, 55.0)
        ldl_cholesterol = st.number_input("LDL Cholesterol", 30.0, 250.0, 120.0)
    with col2:
        triglycerides = st.number_input("Triglycerides", 30.0, 500.0, 140.0)
        glucose_postprandial = st.number_input("Post-meal Glucose", 70.0, 350.0, 140.0)
        insulin_level = st.number_input("Insulin Level", 1.0, 50.0, 10.0)

# Lifestyle
with st.expander("Lifestyle Factors"):
    col1, col2 = st.columns(2)
    with col1:
        physical_activity_minutes_per_week = st.slider("Physical Activity (min/week)", 0, 600, 150)
        diet_score = st.slider("Diet Quality Score", 0, 100, 60)
    with col2:
        sleep_hours_per_day = st.slider("Sleep (hours/day)", 3, 12, 7)
        screen_time_hours_per_day = st.slider("Screen Time (hours/day)", 0, 16, 4)
        heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 72)

# ================================
# AUTOFILL ENGINE (from output.py) with defaults for removed/hid columns
# ================================
DEFAULTS = {
    "Age": 40,
    "gender": "Male",
    # The following demographic columns are hidden in the UI but required by the model.
    # We set sensible defaults here.
    "ethnicity": "Asian",            # default used instead of showing UI control
    "education_level": "Graduate",   # default used instead of showing UI control
    "income_level": "Middle",        # default used instead of showing UI control
    "employment_status": "Employed", # default used instead of showing UI control
    "smoking_status": "Non-smoker",
    "physical_activity_minutes_per_week": 150,
    "diet_score": 60,
    "sleep_hours_per_day": 7,
    "screen_time_hours_per_day": 4,
    "family_history_diabetes": "No",
    "hypertension_history": "No",
    "cardiovascular_history": "No",
    "bmi": 25.0,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "cholesterol_total": 180,
    "hdl_cholesterol": 55,
    "ldl_cholesterol": 120,
    "triglycerides": 140,
    "glucose_fasting": 95,
    "glucose_postprandial": 140,
    "insulin_level": 10.0,
    "hba1c": 5.5,
    "heart_rate": 72,
    "alcohol_consumption_per_week": 0,
    "waist_to_hip_ratio": 0.9
}

def auto_fill(age, gender, ethnicity, bmi, systolic_bp, diastolic_bp):
    row = DEFAULTS.copy()
    row.update({
        "Age": int(age),
        "gender": gender,
        "ethnicity": ethnicity,   # kept for consistency (default or if expanded later)
        "bmi": float(bmi),
        "systolic_bp": int(systolic_bp),
        "diastolic_bp": int(diastolic_bp),
    })
    row["waist_to_hip_ratio"] = round(min(max(0.7 + (bmi - 22) * 0.01, 0.6), 1.5), 2)
    chol = int(160 + age * 0.6 + (bmi - 22) * 1.5)
    row["cholesterol_total"] = int(min(max(chol, 120), 300))
    row["hdl_cholesterol"] = int(min(max(40 + np.random.randint(-4, 8), 20), 100))
    row["ldl_cholesterol"] = int(min(max(100 + (bmi - 22) * 2 + np.random.randint(-5, 15), 30), 250))
    row["triglycerides"] = int(min(max(90 + (bmi - 22) * 3 + np.random.randint(-10, 30), 30), 500))

    gf = int(min(max(85 + (bmi - 22) * 0.8 + np.random.randint(-5, 12), 70), 200))
    gp = int(min(max(110 + (bmi - 22) * 1.3 + np.random.randint(-5, 25), 90), 300))
    row["glucose_fasting"] = gf
    row["glucose_postprandial"] = gp
    row["insulin_level"] = round(min(max(7 + (bmi - 22) * 0.2 + np.random.random(), 2), 50), 1)
    row["hba1c"] = round(min(max(5.0 + (bmi - 22) * 0.03 + np.random.random()*0.2, 4.0), 11.0), 2)
    row["physical_activity_minutes_per_week"] = physical_activity_minutes_per_week if 'physical_activity_minutes_per_week' in globals() else 150
    row["diet_score"] = diet_score if 'diet_score' in globals() else 60
    row["sleep_hours_per_day"] = sleep_hours_per_day if 'sleep_hours_per_day' in globals() else 7
    row["screen_time_hours_per_day"] = screen_time_hours_per_day if 'screen_time_hours_per_day' in globals() else 4
    row["heart_rate"] = int(min(max(70 + (bmi - 22) * 0.5 + np.random.randint(-5, 6), 40), 150))
    row["alcohol_consumption_per_week"] = alcohol_consumption_per_week if 'alcohol_consumption_per_week' in globals() else 0
    row["family_history_diabetes"] = family_history_diabetes if 'family_history_diabetes' in globals() else "No"
    row["hypertension_history"] = hypertension_history if 'hypertension_history' in globals() else "No"
    row["cardiovascular_history"] = cardiovascular_history if 'cardiovascular_history' in globals() else "No"
    # Keep hidden demographics as defaults for model compatibility
    row["education_level"] = DEFAULTS["education_level"]
    row["income_level"] = DEFAULTS["income_level"]
    row["employment_status"] = DEFAULTS["employment_status"]
    row["ethnicity"] = DEFAULTS["ethnicity"]
    row["smoking_status"] = smoking_status if 'smoking_status' in globals() else DEFAULTS["smoking_status"]

    return pd.DataFrame([row])

def risk_level_text(x):
    if x < 20:
        return "Low"
    if x < 40:
        return "Moderate"
    return "High"

# ================================
# PREDICTION BUTTON
# ================================
st.markdown("---")

if st.button("Predict Health Risks"):
    # build input DataFrame using auto_fill - hidden fields get defaults
    input_df = auto_fill(Age, gender, DEFAULTS["ethnicity"], bmi, systolic_bp, diastolic_bp)

    # ensure any UI-provided fields overwrite defaults (lifestyle & labs)
    input_df["physical_activity_minutes_per_week"] = physical_activity_minutes_per_week
    input_df["diet_score"] = diet_score
    input_df["sleep_hours_per_day"] = sleep_hours_per_day
    input_df["screen_time_hours_per_day"] = screen_time_hours_per_day
    input_df["heart_rate"] = heart_rate
    input_df["alcohol_consumption_per_week"] = alcohol_consumption_per_week
    input_df["cholesterol_total"] = cholesterol_total
    input_df["hdl_cholesterol"] = hdl_cholesterol
    input_df["ldl_cholesterol"] = ldl_cholesterol
    input_df["triglycerides"] = triglycerides
    input_df["glucose_postprandial"] = glucose_postprandial
    input_df["insulin_level"] = insulin_level
    input_df["hba1c"] = hba1c
    input_df["glucose_fasting"] = glucose_fasting

    # map yes/no to 1/0 for history flags
    for c in ["family_history_diabetes", "hypertension_history", "cardiovascular_history"]:
        if c in input_df.columns:
            input_df[c] = input_df[c].map({"Yes": 1, "No": 0})

    # ensure smoking_status present and in expected format
    input_df["smoking_status"] = smoking_status if 'smoking_status' in globals() else DEFAULTS["smoking_status"]

    # final safety: ensure all model-required columns exist; if not, add with defaults
    model_expected_cols = pipe.feature_names_in_ if hasattr(pipe, "feature_names_in_") else None
    if model_expected_cols is not None:
        for col in model_expected_cols:
            if col not in input_df.columns:
                # add missing column with a safe default (0 or DEFAULTS entry if present)
                input_df[col] = DEFAULTS.get(col, 0)

    # predict
    preds = pipe.predict(input_df)
    preds = preds[0] if preds.ndim == 2 else np.ravel(preds)

    labels = ["Diabetes", "Hypertension", "Heart Disease", "Obesity", "Cholesterol"]
    results = {labels[i]: float(preds[i]) for i in range(len(labels))}

    cols = st.columns(3)
    idx = 0
    for condition, value in results.items():
        with cols[idx % 3]:
            st.markdown(
                f'<div class="card"><h4>{condition}</h4><div class="value">{value:.2f}%</div><div class="meta">Risk level: {risk_level_text(value)}</div></div>',
                unsafe_allow_html=True,
            )
        idx += 1

    plot_df = pd.DataFrame({"Condition": list(results.keys()), "Risk (%)": list(results.values())})
    fig = px.bar(plot_df, x="Condition", y="Risk (%)", text="Risk (%)",
                 color="Risk (%)", color_continuous_scale=["#60a5fa", "#1e88e5", "#0b5fb5"])
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(template="plotly_dark", yaxis_title="Risk (%)", height=420, margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show auto-filled inputs"):
        st.dataframe(input_df.T, use_container_width=True)
