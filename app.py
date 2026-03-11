import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.title("Titanic Survival Predictor")

pclass = st.selectbox("Passenger Class", [1,2,3])
age = st.number_input("Age")
fare = st.number_input("Fare")

if st.button("Predict"):

    prediction = model.predict([[pclass, age, fare]])

    st.write("Prediction:", prediction)
