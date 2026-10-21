import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess_data
import os

st.set_page_config(page_title="Healthcare No-Show Predictor", page_icon="🏥", layout="centered")

@st.cache_resource
def load_trained_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(current_dir, '../data/kaggle_no_shows.csv'))
    df = load_and_preprocess_data(data_path)
    
    if 'gender' in df.columns:
        df['gender_binary'] = df['gender'].apply(lambda x: 1 if str(x).strip().upper() == 'M' else 0)

    exclude_cols = ['patientid', 'appointmentid', 'scheduledday', 'appointmentday', 'no_show', 'target', 'neighbourhood', 'gender']
    features = [col for col in df.columns if col not in exclude_cols and col != 'target']
    
    X = df[features]
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model, features

model, feature_names = load_trained_model()

st.title("🏥 Medical Appointment No-Show Predictor")
st.markdown("Predict the probability of a patient missing their scheduled medical appointment using Machine Learning.")

st.sidebar.header("Patient Parameters")
age = st.sidebar.slider("Patient Age", 0, 100, 30)
waiting_days = st.sidebar.slider("Waiting Days (Gap between booking & appointment)", 0, 90, 5)
scholarship = st.sidebar.selectbox("Enrolled in Welfare Program (Scholarship)?", [0, 1])
hipertension = st.sidebar.selectbox("Has Hypertension?", [0, 1])
diabetes = st.sidebar.selectbox("Has Diabetes?", [0, 1])
sms_received = st.sidebar.selectbox("Received SMS Reminder?", [0, 1])
gender_binary = st.sidebar.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])

# Build input dataframe matching training features
input_data = {
    'age': age,
    'scholarship': scholarship,
    'hipertension': hipertension,
    'diabetes': diabetes,
    'sms_received': sms_received,
    'waiting_days': waiting_days,
    'appointment_day_of_week': 2, # default mid-week
    'gender_binary': gender_binary
}

# Ensure keys match model features precisely
input_df = pd.DataFrame([input_data])
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

if st.button("Predict No-Show Risk", type="primary"):
    prediction_prob = model.predict_proba(input_df)[0][1]
    
    st.subheader("Prediction Result")
    if prediction_prob > 0.5:
        st.error(f"⚠️ High Risk! Probability of No-Show: **{prediction_prob * 100:.2f}%**")
        st.info("Recommendation: Send an extra automated voice reminder or offer rescheduling options.")
    else:
        st.success(f"✅ Low Risk. Probability of No-Show: **{prediction_prob * 100:.2f}%**")