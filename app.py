import streamlit as st
import pandas as pd
import joblib
import os

# Model ka relative path
model_path = os.path.join(os.path.dirname(__file__), 'churn_model.pkl')
model = joblib.load(model_path)

# App title
st.title("Customer Churn Prediction App")
st.write("Customer ki details bharo aur pata karo ki woh churn karega ya nahi!")

# User se input lo
st.header("Customer Details")

tenure = st.slider(
    "Kitne mahine se customer hai?",
    min_value=1,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges (Rs.)",
    min_value=10.0,
    max_value=150.0,
    value=65.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

# Input ko model ke format mein badlo
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'InternetService_Fiber optic': [1 if internet_service == "Fiber optic" else 0],
    'InternetService_No': [1 if internet_service == "No" else 0],
    'TechSupport_No internet service': [1 if tech_support == "No internet service" else 0],
    'TechSupport_Yes': [1 if tech_support == "Yes" else 0]
})

# Predict button
if st.button("Predict Karo!"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    churn_prob = round(probability[0][1] * 100, 2)
    stay_prob = round(probability[0][0] * 100, 2)

    st.header("Result:")

    if prediction[0] == 1:
        st.error(f"⚠️ Yeh customer CHURN KAREGA!")
    else:
        st.success(f"✅ Yeh customer NAHI jayega!")

    st.write(f"Churn ki probability: **{churn_prob}%**")
    st.write(f"Rukne ki probability: **{stay_prob}%**")