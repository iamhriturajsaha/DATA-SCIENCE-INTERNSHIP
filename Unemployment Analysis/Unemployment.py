# Install Required Libraries
!pip install pandas numpy matplotlib seaborn plotly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Upload Dataset
from google.colab import files
uploaded = files.upload()
df1 = pd.read_csv("1.csv")
df2 = pd.read_csv("2.csv")
print("Dataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)

# Dataset Overview
df1.head()
df2.head()

# Check dataset structure
df1.info()
df2.info()
df1.describe()
df2.describe()

# Check missing values
df1.isnull().sum()
df2.isnull().sum()

# Data Cleaning
# Remove unwanted spaces in column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Convert date column
df1['Date'] = pd.to_datetime(df1['Date'], dayfirst=True)

# Extract year and month
df1['Year'] = df1['Date'].dt.year
df1['Month'] = df1['Date'].dt.month

# Unemployment Rate Distribution
plt.figure(figsize=(8,5))
sns.histplot(df1['Estimated Unemployment Rate (%)'], bins=30, kde=True)
plt.title("Distribution of Unemployment Rate")
plt.show()

# Unemployment Rate Over Time
plt.figure(figsize=(12,6))
sns.lineplot(
    x="Date",
    y="Estimated Unemployment Rate (%)",
    data=df1
)
plt.title("Unemployment Rate Over Time")
plt.show()

# COVID Impact Analysis
covid_data = df1[df1['Year'] == 2020]
plt.figure(figsize=(12,6))
sns.lineplot(
    x="Month",
    y="Estimated Unemployment Rate (%)",
    data=covid_data
)
plt.title("Unemployment Rate During COVID-19 (2020)")
plt.show()

# State-wise Unemployment Analysis
state_unemployment = df1.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean().sort_values(ascending=False)
state_unemployment.plot(
    kind="bar",
    figsize=(12,6),
    color="orange"
)
plt.title("Average Unemployment Rate by State")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# Heatmap Visualization
pivot_table = df1.pivot_table(
    values="Estimated Unemployment Rate (%)",
    index="Region",
    columns="Month"
)
plt.figure(figsize=(12,8))
sns.heatmap(pivot_table, cmap="coolwarm")
plt.title("State vs Month Unemployment Heatmap")
plt.show()

# Interactive Plot
fig = px.line(
    df1,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    color="Region",
    title="Unemployment Rate Trend by Region"
)
fig.show()

# Correlation Analysis
plt.figure(figsize=(6,4))
sns.heatmap(df1.corr(numeric_only=True), annot=True, cmap="viridis")
plt.title("Correlation Matrix")
plt.show()

"""**Key Insights**

• Unemployment increased significantly during 2020 due to COVID-19.

• Some regions experienced much higher unemployment than others.

• There is seasonal variation in unemployment rates.

• The lockdown months show the highest spikes.
"""

# Streamli App
# Install Required Libraries
!pip install streamlit pandas numpy plotly seaborn pyngrok

# Commented out IPython magic to ensure Python compatibility.
# # Create the Streamlit App File
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# st.set_page_config(page_title="Unemployment Analysis Dashboard", page_icon="📊", layout="wide")
# 
# # Load Data
# @st.cache_data
# def load_data():
#     df1 = pd.read_csv("1.csv")
#     df2 = pd.read_csv("2.csv")
#     df1.columns = df1.columns.str.strip()
#     df2.columns = df2.columns.str.strip()
#     df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
#     df1["Year"] = df1["Date"].dt.year
#     df1["Month"] = df1["Date"].dt.month
#     return df1, df2
# df1, df2 = load_data()
# 
# # Sidebar Filters
# st.sidebar.title("Filters")
# region_options = sorted(df1["Region"].dropna().unique())
# year_options = sorted(df1["Year"].dropna().unique())
# regions = st.sidebar.multiselect(
#     "Select Region",
#     options=region_options,
#     default=region_options
# )
# years = st.sidebar.multiselect(
#     "Select Year",
#     options=year_options,
#     default=year_options
# )
# filtered_df = df1[
#     (df1["Region"].isin(regions)) &
#     (df1["Year"].isin(years))
# ]
# if filtered_df.empty:
#     st.warning("No data available for selected filters.")
#     st.stop()
# 
# # Title
# st.title("📊 Unemployment Analysis Dashboard")
# st.markdown("Analysis of unemployment trends and the impact of COVID-19.")
# st.divider()
# 
# # KPI Section
# col1, col2, col3 = st.columns(3)
# avg_unemployment = filtered_df["Estimated Unemployment Rate (%)"].mean()
# max_unemployment = filtered_df["Estimated Unemployment Rate (%)"].max()
# min_unemployment = filtered_df["Estimated Unemployment Rate (%)"].min()
# col1.metric("Average Unemployment Rate", f"{avg_unemployment:.2f}%")
# col2.metric("Maximum Rate", f"{max_unemployment:.2f}%")
# col3.metric("Minimum Rate", f"{min_unemployment:.2f}%")
# st.divider()
# 
# # Unemployment Trend
# st.subheader("📈 Unemployment Trend Over Time")
# fig_trend = px.line(
#     filtered_df,
#     x="Date",
#     y="Estimated Unemployment Rate (%)",
#     color="Region",
#     title="Unemployment Rate by Region"
# )
# st.plotly_chart(fig_trend, use_container_width=True)
# 
# # State Comparison
# st.subheader("📊 State-wise Average Unemployment")
# state_avg = filtered_df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().reset_index()
# fig_state = px.bar(
#     state_avg,
#     x="Region",
#     y="Estimated Unemployment Rate (%)",
#     color="Region",
#     title="Average Unemployment Rate by Region"
# )
# st.plotly_chart(fig_state, use_container_width=True)
# 
# # COVID Impact
# st.subheader("🦠 COVID-19 Impact Analysis (2020)")
# covid_data = filtered_df[filtered_df["Year"] == 2020]
# fig_covid = px.line(
#     covid_data,
#     x="Month",
#     y="Estimated Unemployment Rate (%)",
#     color="Region",
#     title="Unemployment During COVID-19"
# )
# st.plotly_chart(fig_covid, use_container_width=True)
# 
# # Data Table
# st.subheader("📄 Dataset Preview")
# st.dataframe(filtered_df)
# 
# # Insights Section
# st.subheader("📌 Key Insights")
# st.markdown("""
# • Unemployment rates increased significantly during 2020 due to COVID-19.
# 
# • Some regions experienced much higher unemployment compared to others.
# 
# • Lockdown months show the largest spikes in unemployment.
# 
# • Economic recovery can be observed in later months.
# """)

# Launch Streamlit
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
