import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay

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

# Features used by model
X = df[['pclass','age','fare']]
y = df['survived']

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1,2,3])
age = st.sidebar.slider("Age", 1, 80, 25)
fare = st.sidebar.slider("Fare", 0, 200, 50)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Charts", "Model Insights"])

# =============================
# TAB 1 — Prediction
# =============================
with tab1:

    st.subheader("Prediction")

    if st.button("Predict Survival"):

        prediction = model.predict([[pclass, age, fare]])
        prob = model.predict_proba([[pclass, age, fare]])

        if prediction[0] == 1:
            st.success("Passenger is likely to SURVIVE")
        else:
            st.error("Passenger is likely to NOT survive")

        st.write("Survival Probability:", round(prob[0][1],2))
        st.progress(float(prob[0][1]))

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

# =============================
# TAB 2 — Charts
# =============================
with tab2:

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
        df['pclass'].value_counts().plot(kind='bar', ax=ax)

        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # Age Distribution
    with col3:
        st.write("Age Distribution")

        fig, ax = plt.subplots()
        df['age'].hist(bins=20, ax=ax)

        st.pyplot(fig)

    # Fare Distribution
    with col4:
        st.write("Fare Distribution")

        fig, ax = plt.subplots()
        df['fare'].hist(bins=20, ax=ax)

        st.pyplot(fig)

    # Survival by Class
    st.subheader("Survival by Passenger Class")

    fig, ax = plt.subplots()
    pd.crosstab(df['pclass'], df['survived']).plot(kind='bar', ax=ax)

    st.pyplot(fig)

# =============================
# TAB 3 — Model Insights
# =============================
with tab3:

    st.subheader("Feature Importance")

    features = ['pclass','age','fare']
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    ax.bar(features, importance)

    st.pyplot(fig)

    st.subheader("Decision Tree Model")

    fig, ax = plt.subplots(figsize=(14,6))

    plot_tree(
        model,
        feature_names=['pclass','age','fare'],
        class_names=["Dead","Survived"],
        filled=True
    )

    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_estimator(
        model,
        X,
        y,
        ax=ax
    )

    st.pyplot(fig)
