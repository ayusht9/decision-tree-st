import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Decision Tree Predictor")

age = st.number_input("Age")
fare = st.number_input("Fare")

if st.button("Predict"):

    prediction = model.predict([[age, fare]])
    st.success(f"Prediction: {prediction[0]}")
