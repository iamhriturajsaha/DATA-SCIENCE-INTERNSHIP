import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load model
model = pickle.load(open("sales_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.set_page_config(
    page_title="AI Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# CSS Styling
st.markdown("""
<style>
.title {
font-size:70px;
font-weight:800;
text-align:center;
color:#2E86C1;
}
.subtitle {
text-align:center;
font-size:22px;
color:gray;
margin-bottom:20px;
}
.metric-card {
background-color:#F4F6F7;
padding:20px;
border-radius:12px;
text-align:center;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="title">📈 AI Sales Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict product sales based on advertising spend</p>', unsafe_allow_html=True)
st.divider()

# Sidebar Inputs
st.sidebar.header("Advertising Budget")
tv = st.sidebar.slider("TV Advertising Budget ($)",0,300,100)
radio = st.sidebar.slider("Radio Advertising Budget ($)",0,50,25)
news = st.sidebar.slider("Newspaper Advertising Budget ($)",0,120,20)

# Prediction
input_data = np.array([[tv,radio,news]])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
st.subheader("📊 Predicted Sales")
st.success(f"Estimated Sales: {prediction[0]:.2f} units")
st.divider()

# Dataset Analytics
st.subheader("Advertising Impact Analysis")
df = pd.read_csv("Advertising.csv")
fig = px.scatter(
    df,
    x="TV",
    y="Sales",
    size="Radio",
    color="Sales",
    title="TV Advertising vs Sales"
)
st.plotly_chart(fig,use_container_width=True)
fig2 = px.scatter(
    df,
    x="Radio",
    y="Sales",
    color="Sales",
    title="Radio Advertising vs Sales"
)
st.plotly_chart(fig2,use_container_width=True)
st.divider()
st.markdown("""
### 🤖 Model Information

Model Used: **Random Forest Regressor**

Features Used:
- TV Advertising Budget
- Radio Advertising Budget
- Newspaper Advertising Budget

Purpose:
Predict product sales based on advertising investment.
""")
