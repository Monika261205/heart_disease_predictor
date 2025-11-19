import streamlit as st
import numpy as np
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Predictor")

# ------  INPUT WIDGETS  ------

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=45)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
sbp = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
dbp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=80)
hr = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
blood_sugar = st.number_input("Blood Sugar (mg/dl)", min_value=50, max_value=400, value=90)

previous_heart_attack = st.selectbox("Previous Heart Attack", ["No", "Yes"])
hyperlipidemia = st.selectbox("Hyperlipidemia (High Cholesterol)", ["No", "Yes"])

# ------  CATEGORY ENCODING ------

gender_map = {"Male": 1, "Female": 0}
yes_no_map = {"No": 0, "Yes": 1}

# numeric values
gender_val = gender_map[gender]
previous_heart_val = yes_no_map[previous_heart_attack]
hyperlipidemia_val = yes_no_map[hyperlipidemia]

# ------  MAKE INPUT ROW ------

input_data = np.array([
    age,
    gender_val,
    weight,
    height,
    sbp,
    dbp,
    hr,
    blood_sugar,
    previous_heart_val,
    hyperlipidemia_val
], dtype=float).reshape(1, -1)

# ------  PREDICTION ------

if st.button("Predict Heart Disease Risk"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 High Risk of Heart Disease (Probability: {prob:.2f})")
        else:
            st.success(f"💚 Low Risk of Heart Disease (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"Error: {str(e)}")
