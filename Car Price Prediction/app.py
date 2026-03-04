import streamlit as st
import numpy as np
import pickle
model = pickle.load(open("car_model.pkl","rb"))
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)
st.markdown("""
<style>
.big-title {
    font-size:60px;
    font-weight:800;
    text-align:center;
    color:#1F618D;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-title">🚗 AI Car Price Prediction</p>', unsafe_allow_html=True)
st.write("Predict the resale price of a car using Machine Learning.")
st.divider()

# Layout
col1,col2 = st.columns(2)
with col1:
    st.markdown("### Car Details")
    present_price = st.slider("Present Price (Lakhs)",0.0,50.0,5.0)
    kms_driven = st.slider("Kilometers Driven",0,200000,50000)
    owner = st.selectbox("Number of Owners",[0,1,2,3])
    car_age = st.slider("Car Age (Years)",0,20,5)
with col2:
    st.markdown("### Car Specifications")
    fuel = st.selectbox(
        "Fuel Type",
        ["Petrol","Diesel","CNG"]
    )
    seller = st.selectbox(
        "Seller Type",
        ["Dealer","Individual"]
    )
    transmission = st.selectbox(
        "Transmission",
        ["Manual","Automatic"]
    )

# Encoding
fuel_map = {"Petrol":2,"Diesel":1,"CNG":0}
seller_map = {"Dealer":0,"Individual":1}
trans_map = {"Manual":1,"Automatic":0}
fuel = fuel_map[fuel]
seller = seller_map[seller]
transmission = trans_map[transmission]

# Prediction
if st.button("Predict Car Price"):
    input_data = np.array([[present_price,kms_driven,owner,fuel,seller,transmission,car_age]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")
st.divider()
st.markdown("""
### Model Information

Model Used: **XGBoost Regressor**

Features Used:

• Present Price  
• Kms Driven  
• Owner  
• Fuel Type  
• Seller Type  
• Transmission  
• Car Age
""")
