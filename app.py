import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Titanic ML Dashboard", layout="wide")

st.title("🚢 Titanic Survival Prediction Using Decision Tree")

# -----------------------------
# Load Data & Model
# -----------------------------
df = pd.read_csv("titanic.csv")
model = joblib.load("model.pkl")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1,2,3])
age = st.sidebar.slider("Age", 1, 80, 25)
fare = st.sidebar.slider("Fare", 0, 200, 50)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Prediction")

if st.button("Predict Survival"):

    prediction = model.predict([[pclass, age, fare]])

    if prediction[0] == 1:
        st.success("Passenger is likely to SURVIVE")
    else:
        st.error("Passenger is likely to NOT survive")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head())

# -----------------------------
# Charts Section
# -----------------------------
st.subheader("Data Visualizations")

col1, col2 = st.columns(2)

# Survival Count
with col1:
    st.write("Survival Count")
    fig, ax = plt.subplots()
    df['survived'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Survived")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Passenger Class
with col2:
    st.write("Passenger Class Distribution")
    fig, ax = plt.subplots()
    df['Pclass'].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

# -----------------------------
# Age Distribution
# -----------------------------
col3, col4 = st.columns(2)

with col3:
    st.write("Age Distribution")
    fig, ax = plt.subplots()
    df['Age'].hist(bins=20, ax=ax)
    st.pyplot(fig)

with col4:
    st.write("Fare Distribution")
    fig, ax = plt.subplots()
    df['Fare'].hist(bins=20, ax=ax)
    st.pyplot(fig)

# -----------------------------
# Survival by Class
# -----------------------------
st.subheader("Survival by Passenger Class")

fig, ax = plt.subplots()

pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=ax)

st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")

features = ['Pclass','Age','Fare']
importance = model.feature_importances_

fig, ax = plt.subplots()

ax.bar(features, importance)

st.pyplot(fig)

# -----------------------------
# Decision Tree Visualization
# -----------------------------
st.subheader("Decision Tree Model")

fig, ax = plt.subplots(figsize=(14,6))

plot_tree(
    model,
    feature_names=['Pclass','Age','Fare'],
    class_names=["Dead","Survived"],
    filled=True
)

st.pyplot(fig)

