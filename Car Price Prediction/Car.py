# Install Libraries
!pip install pandas numpy scikit-learn streamlit pyngrok
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Upload Dataset
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("Car.csv")
df.head()

# Data Cleaning
df = df.drop(columns=["Car_Name"])

# Convert categorical variables
df["Fuel_Type"] = encoder.fit_transform(df["Fuel_Type"])
df["Selling_type"] = encoder.fit_transform(df["Selling_type"])
df["Transmission"] = encoder.fit_transform(df["Transmission"])

# Feature engineering
df["Car_Age"] = 2024 - df["Year"]
df = df.drop(columns=["Year"])

# Define Features and Target
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

# Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=200)
model.fit(X_train,y_train)

# Model Evaluation
pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test,pred))
print("MAE:", mean_absolute_error(y_test,pred))

# Save Model
pickle.dump(model,open("car_model.pkl","wb"))

# Commented out IPython magic to ensure Python compatibility.
# # Streamlit App
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# model = pickle.load(open("car_model.pkl","rb"))
# st.set_page_config(
#     page_title="Car Price Predictor",
#     page_icon="🚗",
#     layout="wide"
# )
# st.title("🚗 Car Price Prediction Dashboard")
# st.markdown(
# """
# Predict the **selling price of a car** using Machine Learning.
# """
# )
# st.divider()
# st.sidebar.header("Enter Car Features")
# present_price = st.sidebar.slider("Present Price (Lakhs)",0.0,50.0,5.0)
# kms_driven = st.sidebar.slider("Kilometers Driven",0,200000,50000)
# owner = st.sidebar.selectbox("Number of Previous Owners",[0,1,2,3])
# fuel = st.sidebar.selectbox(
#     "Fuel Type",
#     ["Petrol","Diesel","CNG"]
# )
# seller = st.sidebar.selectbox(
#     "Seller Type",
#     ["Dealer","Individual"]
# )
# transmission = st.sidebar.selectbox(
#     "Transmission",
#     ["Manual","Automatic"]
# )
# car_age = st.sidebar.slider("Car Age (Years)",0,20,5)
# fuel_map = {"Petrol":2,"Diesel":1,"CNG":0}
# seller_map = {"Dealer":0,"Individual":1}
# trans_map = {"Manual":1,"Automatic":0}
# fuel = fuel_map[fuel]
# seller = seller_map[seller]
# transmission = trans_map[transmission]
# input_data = np.array([[present_price,kms_driven,owner,fuel,seller,transmission,car_age]])
# prediction = model.predict(input_data)
# col1,col2 = st.columns(2)
# with col1:
#     st.subheader("Input Features")
#     st.write({
#         "Present Price":present_price,
#         "Kms Driven":kms_driven,
#         "Owners":owner,
#         "Fuel":fuel,
#         "Seller":seller,
#         "Transmission":transmission,
#         "Car Age":car_age
#     })
# with col2:
#     st.subheader("Predicted Price")
#     st.success(f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")
# st.divider()
# st.markdown(
# """
# ### About Model
# 
# Model Used: Random Forest Regressor
# 
# Features Used:
# - Present Price
# - Kms Driven
# - Fuel Type
# - Seller Type
# - Transmission
# - Car Age
# """
# )

# Run Streamlit
from pyngrok import ngrok
import time
import os
ngrok.kill()
!streamlit run app.py --server.port 8501 --server.headless true &>/content/logs.txt &
time.sleep(5)
public_url = ngrok.connect(8501)
print("🌍 Open this URL:", public_url)
