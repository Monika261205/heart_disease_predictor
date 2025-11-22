import streamlit as st # type: ignore
import pandas as pd
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from joblib import load

# =============================
#  LOAD MODEL & PREPROCESSORS
# =============================
model = load("final_heart_disease_model.pkl")
encoder = load("encoder.pkl")
scaler = load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Fill in the patient details below to predict the risk of heart disease.")

# =============================
#  USER INPUT FIELDS
# =============================
age = st.number_input("Age", 1, 120, 45)
gender = st.selectbox("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)", 20, 200, 70)
height = st.number_input("Height (cm)", 100, 230, 170)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
alcohol = st.selectbox("Alcohol Intake", ["None", "Low", "Moderate", "High"])
physical = st.selectbox("Physical Activity", ["Sedentary", "Moderate", "Active"])
diet = st.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])
stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])

hypertension = st.selectbox("Hypertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
hyperlipidemia = st.selectbox("Hyperlipidemia", [0, 1])
family_history = st.selectbox("Family History", [0, 1])
previous_attack = st.selectbox("Previous Heart Attack", [0, 1])

systolic = st.number_input("Systolic BP", 80, 250, 120)
diastolic = st.number_input("Diastolic BP", 40, 150, 80)
heart_rate = st.number_input("Heart Rate", 40, 200, 72)
blood_sugar = st.number_input("Blood Sugar (Fasting)", 50, 300, 100)
cholesterol = st.number_input("Cholesterol Total", 80, 400, 200)

# =============================
#  PREDICTION
# =============================
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Smoking": smoking,
        "Alcohol_Intake": alcohol,
        "Physical_Activity": physical,
        "Diet": diet,
        "Stress_Level": stress,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Hyperlipidemia": hyperlipidemia,
        "Family_History": family_history,
        "Previous_Heart_Attack": previous_attack,
        "Systolic_BP": systolic,
        "Diastolic_BP": diastolic,
        "Heart_Rate": heart_rate,
        "Blood_Sugar_Fasting": blood_sugar,
        "Cholesterol_Total": cholesterol
    }])

    # Encode + scale like training
    encoded = encoder.transform(input_data)
    scaled = scaler.transform(encoded)

    prediction = model.predict(scaled)[0]

    # Show prediction result
    if prediction == 1:
        st.error("⚠ The model predicts HIGH risk of heart disease.")
    else:
        st.success("✔ The model predicts LOW risk of heart disease.")

    # =====================================================
    #     VISUAL INSIGHTS SECTION
    # =====================================================
    st.subheader("📊 Visual Insights")

    # -------------------------
    # 1️⃣ USER INPUT BAR CHART
    # -------------------------
    display_df = input_data.copy()
    display_numeric = display_df.select_dtypes(include=["int64", "float64"]).iloc[0]

    fig_user = px.bar(
        x=display_numeric.index,
        y=display_numeric.values,
        title="Your Health Measurements",
        labels={"x": "Feature", "y": "Value"}
    )
    st.plotly_chart(fig_user, use_container_width=True)

    # -------------------------
    # 2️⃣ RISK GAUGE
    # -------------------------
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(prediction),
        gauge={
            "axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, 0.5], "color": "lightgreen"},
                {"range": [0.5, 1], "color": "salmon"}
            ]
        },
        title={"text": "Heart Disease Risk Level"}
    ))

    st.plotly_chart(fig_gauge, use_container_width=True)

    # -------------------------
    # 3️⃣ FEATURE IMPORTANCE
    # -------------------------
    if hasattr(model, "feature_importances_"):
        try:
            importance_df = pd.DataFrame({
                "Feature": encoder.get_feature_names_out(),
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig_imp = px.bar(
                importance_df.head(10),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Model Feature Importances"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except:
            st.info("Feature importance could not be displayed due to model encoding format.")
    else:
        st.info("Feature importance is not available for this model.")
