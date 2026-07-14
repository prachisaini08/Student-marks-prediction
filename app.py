import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Student Marks Predictor", layout="wide")

st.title("🎓 Student Marks Predictor")
st.subheader("Linear Regression Model")

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("student_marks.csv")

X = df[['Hours']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("About Project")

st.sidebar.write("""
This project predicts student marks using
**Linear Regression**.

Dataset Features:
- Hours
- Marks
""")

# ------------------------------
# Layout
# ------------------------------
col1, col2 = st.columns(2)

with col1:

    st.header("Input")

    hours = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=24.0,
        value=5.0
    )

    if st.button("Predict Marks"):

        prediction = model.predict([[hours]])

        st.success(f"Predicted Marks = {prediction[0]:.2f}")

with col2:

    st.header("Regression Graph")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(X, y, color="blue", label="Actual Data")

    ax.plot(
        X,
        model.predict(X),
        color="red",
        linewidth=3,
        label="Regression Line"
    )

    ax.set_xlabel("Hours")
    ax.set_ylabel("Marks")

    ax.legend()

    st.pyplot(fig)

# ------------------------------
# Model Details
# ------------------------------

st.markdown("---")

col3,col4,col5=st.columns(3)

with col3:
    st.metric("Slope", round(model.coef_[0],2))

with col4:
    st.metric("Intercept", round(model.intercept_,2))

with col5:
    st.metric("Score (R²)", round(model.score(X,y),2))

st.markdown("---")

st.write(df)