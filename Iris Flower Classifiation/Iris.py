# Install and Import Libraries
!pip install seaborn scikit-learn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("Iris.csv")
df.head()

# Dataset Overview
df.shape
df.info()
df.describe()

# Check missing values
df.isnull().sum()

# Exploratory Data Analysis (EDA)
# Species Distribution
sns.countplot(x='Species', data=df)
plt.title("Species Distribution")
plt.show()

# Pairplot
sns.pairplot(df, hue='Species')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.drop(columns=['Id','Species']).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# Data Preprocessing
# Remove unnecessary column
df = df.drop(columns=['Id'])

# Encode species labels
encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])

# Separate features and target
X = df.drop(columns=['Species'])
y = df['Species']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Machine Learning Models
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))

# Model Evaluation
# Classification Report
print(classification_report(y_test, pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, pred_rf)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Test With Custom Input
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=X.columns
    )
    prediction = rf.predict(input_data)
    species = encoder.inverse_transform(prediction)
    print("Predicted Flower:", species[0])
predict_flower(5.1,3.5,1.4,0.2)

# Feature Importance
importance = rf.feature_importances_
features = X.columns
plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()

# Streamlit App
# Install Required Libraries
!pip install streamlit pyngrok scikit-learn seaborn

# Train and Save Model
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])
encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])
X = df.drop(columns=["Species"])
y = df["Species"]
model = RandomForestClassifier(n_estimators=100)
model.fit(X,y)
pickle.dump(model,open("iris_model.pkl","wb"))
pickle.dump(encoder,open("encoder.pkl","wb"))
print("Model Saved")

# Commented out IPython magic to ensure Python compatibility.
# # Create a file called app.py
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# 
# # Load model
# model = pickle.load(open("iris_model.pkl","rb"))
# encoder = pickle.load(open("encoder.pkl","rb"))
# st.set_page_config(
#     page_title="Iris Flower Classifier",
#     page_icon="🌸",
#     layout="wide"
# )
# 
# # Title
# st.title("🌸 Iris Flower Classification App")
# st.markdown("Predict Iris flower species using Machine Learning.")
# st.divider()
# 
# # Sidebar
# st.sidebar.header("Input Flower Measurements")
# sepal_length = st.sidebar.slider("Sepal Length (cm)",4.0,8.0,5.1)
# sepal_width = st.sidebar.slider("Sepal Width (cm)",2.0,4.5,3.5)
# petal_length = st.sidebar.slider("Petal Length (cm)",1.0,7.0,1.4)
# petal_width = st.sidebar.slider("Petal Width (cm)",0.1,2.5,0.2)
# input_data = pd.DataFrame({
#     "SepalLengthCm":[sepal_length],
#     "SepalWidthCm":[sepal_width],
#     "PetalLengthCm":[petal_length],
#     "PetalWidthCm":[petal_width]
# })
# 
# # Layout columns
# col1,col2 = st.columns(2)
# with col1:
#     st.subheader("Input Features")
#     st.dataframe(input_data)
# with col2:
#     st.subheader("Prediction")
#     prediction = model.predict(input_data)
#     species = encoder.inverse_transform(prediction)
#     st.success(f"Predicted Species: {species[0]}")
# st.divider()
# 
# # Visualization
# st.subheader("Feature Values Visualization")
# chart_data = pd.DataFrame({
#     "Feature":["Sepal Length","Sepal Width","Petal Length","Petal Width"],
#     "Value":[sepal_length,sepal_width,petal_length,petal_width]
# })
# st.bar_chart(chart_data.set_index("Feature"))
# st.divider()
# st.markdown(
# """
# ### About
# This application predicts Iris flower species using a Random Forest Machine Learning model trained on the famous Iris dataset.
# 
# **Features Used**
# - Sepal Length
# - Sepal Width
# - Petal Length
# - Petal Width
# """
# )

# Launch Streamlit
!pip install streamlit pyngrok
!ngrok config add-authtoken 2z0Oqv0tD166fELGCHwV2gLZwq1_2G2zUQRSs6C27k9vdzxwq
from pyngrok import ngrok
import time
import os
# Kill existing tunnels
ngrok.kill()
# Start Streamlit server
!streamlit run app.py --server.port 8501 --server.headless true &>/content/logs.txt &
# Wait for server to start
time.sleep(5)
# Create public tunnel
public_url = ngrok.connect(8501)
print("🌍 Open this URL:", public_url)
