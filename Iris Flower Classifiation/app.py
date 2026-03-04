import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("iris_model.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Title
st.title("🌸 Iris Flower Classification App")
st.markdown("Predict Iris flower species using Machine Learning.")
st.divider()

# Sidebar
st.sidebar.header("Input Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)",4.0,8.0,5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)",2.0,4.5,3.5)
petal_length = st.sidebar.slider("Petal Length (cm)",1.0,7.0,1.4)
petal_width = st.sidebar.slider("Petal Width (cm)",0.1,2.5,0.2)
input_data = pd.DataFrame({
    "SepalLengthCm":[sepal_length],
    "SepalWidthCm":[sepal_width],
    "PetalLengthCm":[petal_length],
    "PetalWidthCm":[petal_width]
})

# Layout columns
col1,col2 = st.columns(2)
with col1:
    st.subheader("Input Features")
    st.dataframe(input_data)
with col2:
    st.subheader("Prediction")
    prediction = model.predict(input_data)
    species = encoder.inverse_transform(prediction)
    st.success(f"Predicted Species: {species[0]}")
st.divider()

# Visualization
st.subheader("Feature Values Visualization")
chart_data = pd.DataFrame({
    "Feature":["Sepal Length","Sepal Width","Petal Length","Petal Width"],
    "Value":[sepal_length,sepal_width,petal_length,petal_width]
})
st.bar_chart(chart_data.set_index("Feature"))
st.divider()
st.markdown(
"""
### About
This application predicts Iris flower species using a Random Forest Machine Learning model trained on the famous Iris dataset.

**Features Used**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
"""
)
