import streamlit as st
import numpy as np
import joblib

# Load model & scaler (same folder)
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Prediction")
st.write("Passenger details fill pannunga:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, value=30.0)

if st.button("Predict"):
    input_data = np.array([[pclass, age, fare]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
