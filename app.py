import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, encoder, scaler
model = joblib.load("final_heart_disease_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient details below:")

# -------------------------------
# INPUT FORM
# -------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=40)
gender = st.selectbox("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Intake", ["Yes", "No"])
diet = st.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])
physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
cholesterol = st.number_input("Cholesterol Total (mg/dL)", min_value=100, max_value=400, value=180)
systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
blood_sugar_fasting = st.number_input("Blood Sugar Fasting (mg/dL)", min_value=60, max_value=300, value=90)

# NEW MISSING FIELDS
previous_heart_attack = st.selectbox(
    "Previous Heart Attack",
    ["Yes", "No"]
)

hyperlipidemia = st.selectbox(
    "Hyperlipidemia (High Cholesterol)",
    ["Yes", "No"]
)

# -------------------------------
# CREATE INPUT DICT
# -------------------------------
user_data = {
    "Age": age,
    "Gender": gender,
    "Weight": weight,
    "Height": height,
    "BMI": bmi,
    "Smoking": smoking,
    "Alcohol_Intake": alcohol,
    "Diet": diet,
    "Physical_Activity": physical_activity,
    "Stress_Level": stress,
    "Hypertension": hypertension,
    "Diabetes": diabetes,
    "Family_History": family_history,
    "Cholesterol_Total": cholesterol,
    "Systolic_BP": systolic_bp,
    "Diastolic_BP": diastolic_bp,
    "Heart_Rate": heart_rate,
    "Blood_Sugar_Fasting": blood_sugar_fasting,
    
    # NEW FIELDS ADDED HERE
    "Previous_Heart_Attack": previous_heart_attack,
    "Hyperlipidemia": hyperlipidemia
}

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Heart Disease Risk"):
    try:
        encoded = encoder.transform(input_df)
        scaled = scaler.transform(encoded)
        prediction = model.predict(scaled)[0]
        
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
            
    except Exception as e:
        st.error(f"Error: {e}")

