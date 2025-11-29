# Health Risk Predictor from Lifestyle & Medical History
<div align="center">

<img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" />
<img src="https://img.shields.io/badge/Model-LightGBM-green?logo=lightgbm" />
<img src="https://img.shields.io/badge/Explainability-SHAP-yellow?logo=python" />
<img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative" />

</div>

---

## Overview

This application is a machine learning–powered health assessment tool that predicts an individual’s risk for multiple chronic conditions using lifestyle, demographic, and medical data. It processes user inputs—such as age, BMI, blood pressure, glucose levels, cholesterol levels, and lifestyle habits—and generates personalized risk scores. The system helps users understand potential health issues early, enabling them to take timely preventive actions.

---

## Live Demo

**Deployed Application:**  
[HealthApp – Streamlit App]("https://healthriskpredictionapp.streamlit.app/")

The Streamlit application allows users to:
- Collects user information including demographics, lifestyle habits, vitals, and medical history.
- Automatically fills missing clinical values using an intelligent autofill engine.
- Processes and prepares the data for machine learning prediction.
- Predicts four health risks at once: Hypertension, Obesity, Heart Disease, and Cholesterol imbalance.
- Displays clear risk percentages with visual charts and easy-to-read result cards.

---

## Repository Structure

```bash

HEALTH APP/
│
├── app.py                     # Main application file (runs the model & API/UI logic)
├── best_pipeline.joblib       # Saved ML model pipeline (trained + optimized)
├── requirements.txt           # Python dependencies required to run the project
├── README.md                  # Project documentation
│
└── .devcontainer/
    └── devcontainer.json      # VS Code Dev Container setup for reproducible environment

