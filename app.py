import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load("final_heart_disease_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the details below:")

# --------------------------------
# INPUT FIELDS
# --------------------------------
age = st.number_input("Age", 1, 120, 40)
gender = st.selectbox("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)", 20, 200, 70)
height = st.number_input("Height (cm)", 100, 220, 170)
bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
smoking = st.selectbox("Smoking", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Intake", ["Yes", "No"])
diet = st.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])
physical_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])
family_history = st.selectbox("Family History", ["Yes", "No"])
cholesterol = st.number_input("Cholesterol Total (mg/dL)", 100, 400, 180)
systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80)
heart_rate = st.number_input("Heart Rate", 40, 200, 75)
blood_sugar_fasting = st.number_input("Blood Sugar Fasting (mg/dL)", 60, 300, 90)
previous_heart_attack = st.selectbox("Previous Heart Attack", ["Yes", "No"])
hyperlipidemia = st.selectbox("Hyperlipidemia (High Cholesterol)", ["Yes", "No"])

# --------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------
input_data = pd.DataFrame([{
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
    "Previous_Heart_Attack": previous_heart_attack,
    "Hyperlipidemia": hyperlipidemia,
}])

# YES/NO → 1/0
binary_cols = [
    "Smoking", "Alcohol_Intake", "Hypertension", "Diabetes",
    "Family_History", "Previous_Heart_Attack", "Hyperlipidemia"
]

for col in binary_cols:
    input_data[col] = input_data[col].map({"Yes": 1, "No": 0})

# --------------------------------
# PREDICT BUTTON
# --------------------------------
if st.button("Predict Heart Disease Risk"):
    try:
        # Make input columns match encoder's expected columns
        expected_cols = encoder.feature_names_in_

        # Add missing columns
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder correctly
        input_data = input_data[expected_cols]

        # Apply encoding & scaling
        encoded = encoder.transform(input_data)
        scaled = scaler.transform(encoded)

        # Model prediction
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as e:
        st.error(f"Error: {e}")
