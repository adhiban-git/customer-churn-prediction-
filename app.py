import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Customer Churn Prediction Dashboard")

st.write("Enter customer details below:")

customer_id = st.number_input("Customer ID", min_value=1, value=1001)
age = st.slider("Age", 18, 100, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

tenure = st.slider("Tenure (Months)", 0, 120, 12)

usage_frequency = st.slider("Usage Frequency", 1, 100, 20)

support_calls = st.slider("Support Calls", 0, 20, 2)

payment_delay = st.slider("Payment Delay (Days)", 0, 60, 5)

subscription_type = st.selectbox(
    "Subscription Type",
    ["Basic", "Standard", "Premium"]
)

contract_length = st.selectbox(
    "Contract Length",
    ["Monthly", "Quarterly", "Yearly"]
)

total_spend = st.number_input(
    "Total Spend",
    min_value=0.0,
    value=1000.0
)

last_interaction = st.slider(
    "Last Interaction (Days Ago)",
    0,
    365,
    30
)

# Encode categorical values
gender_map = {
    "Male": 1,
    "Female": 0
}

subscription_map = {
    "Basic": 0,
    "Standard": 1,
    "Premium": 2
}

contract_map = {
    "Monthly": 0,
    "Quarterly": 1,
    "Yearly": 2
}

if st.button("Predict Churn"):

    data = pd.DataFrame({
        "CustomerID": [customer_id],
        "Age": [age],
        "Gender": [gender_map[gender]],
        "Tenure": [tenure],
        "Usage Frequency": [usage_frequency],
        "Support Calls": [support_calls],
        "Payment Delay": [payment_delay],
        "Subscription Type": [subscription_map[subscription_type]],
        "Contract Length": [contract_map[contract_length]],
        "Total Spend": [total_spend],
        "Last Interaction": [last_interaction]
    })

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    if prediction == 1:
        st.error("⚠ Customer is likely to churn")
        st.write(f"Churn Probability: {probability[1]*100:.2f}%")
    else:
        st.success("✅ Customer is likely to stay")
        st.write(f"Retention Probability: {probability[0]*100:.2f}%")