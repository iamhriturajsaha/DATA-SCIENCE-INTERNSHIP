import streamlit as st
import pandas as pd
import numpy as np
import pickle
model = pickle.load(open("car_model.pkl","rb"))
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)
st.title("🚗 Car Price Prediction Dashboard")
st.markdown(
"""
Predict the **selling price of a car** using Machine Learning.
"""
)
st.divider()
st.sidebar.header("Enter Car Features")
present_price = st.sidebar.slider("Present Price (Lakhs)",0.0,50.0,5.0)
kms_driven = st.sidebar.slider("Kilometers Driven",0,200000,50000)
owner = st.sidebar.selectbox("Number of Previous Owners",[0,1,2,3])
fuel = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol","Diesel","CNG"]
)
seller = st.sidebar.selectbox(
    "Seller Type",
    ["Dealer","Individual"]
)
transmission = st.sidebar.selectbox(
    "Transmission",
    ["Manual","Automatic"]
)
car_age = st.sidebar.slider("Car Age (Years)",0,20,5)
fuel_map = {"Petrol":2,"Diesel":1,"CNG":0}
seller_map = {"Dealer":0,"Individual":1}
trans_map = {"Manual":1,"Automatic":0}
fuel = fuel_map[fuel]
seller = seller_map[seller]
transmission = trans_map[transmission]
input_data = np.array([[present_price,kms_driven,owner,fuel,seller,transmission,car_age]])
prediction = model.predict(input_data)
col1,col2 = st.columns(2)
with col1:
    st.subheader("Input Features")
    st.write({
        "Present Price":present_price,
        "Kms Driven":kms_driven,
        "Owners":owner,
        "Fuel":fuel,
        "Seller":seller,
        "Transmission":transmission,
        "Car Age":car_age
    })
with col2:
    st.subheader("Predicted Price")
    st.success(f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")
st.divider()
st.markdown(
"""
### About Model

Model Used: Random Forest Regressor

Features Used:
- Present Price
- Kms Driven
- Fuel Type
- Seller Type
- Transmission
- Car Age
"""
)
