import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="Unemployment Analysis Dashboard", page_icon="📊", layout="wide")

# Load Data
@st.cache_data
def load_data():
    df1 = pd.read_csv("1.csv")
    df2 = pd.read_csv("2.csv")
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df1["Date"] = pd.to_datetime(df1["Date"], dayfirst=True)
    df1["Year"] = df1["Date"].dt.year
    df1["Month"] = df1["Date"].dt.month
    return df1, df2
df1, df2 = load_data()

# Sidebar Filters
st.sidebar.title("Filters")
region_options = sorted(df1["Region"].dropna().unique())
year_options = sorted(df1["Year"].dropna().unique())
regions = st.sidebar.multiselect(
    "Select Region",
    options=region_options,
    default=region_options
)
years = st.sidebar.multiselect(
    "Select Year",
    options=year_options,
    default=year_options
)
filtered_df = df1[
    (df1["Region"].isin(regions)) &
    (df1["Year"].isin(years))
]
if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()
    
# Title
st.title("📊 Unemployment Analysis Dashboard")
st.markdown("Analysis of unemployment trends and the impact of COVID-19.")
st.divider()

# KPI Section
col1, col2, col3 = st.columns(3)
avg_unemployment = filtered_df["Estimated Unemployment Rate (%)"].mean()
max_unemployment = filtered_df["Estimated Unemployment Rate (%)"].max()
min_unemployment = filtered_df["Estimated Unemployment Rate (%)"].min()
col1.metric("Average Unemployment Rate", f"{avg_unemployment:.2f}%")
col2.metric("Maximum Rate", f"{max_unemployment:.2f}%")
col3.metric("Minimum Rate", f"{min_unemployment:.2f}%")
st.divider()

# Unemployment Trend
st.subheader("📈 Unemployment Trend Over Time")
fig_trend = px.line(
    filtered_df,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    color="Region",
    title="Unemployment Rate by Region"
)
st.plotly_chart(fig_trend, use_container_width=True)

# State Comparison
st.subheader("📊 State-wise Average Unemployment")
state_avg = filtered_df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().reset_index()
fig_state = px.bar(
    state_avg,
    x="Region",
    y="Estimated Unemployment Rate (%)",
    color="Region",
    title="Average Unemployment Rate by Region"
)
st.plotly_chart(fig_state, use_container_width=True)

# COVID Impact
st.subheader("🦠 COVID-19 Impact Analysis (2020)")
covid_data = filtered_df[filtered_df["Year"] == 2020]
fig_covid = px.line(
    covid_data,
    x="Month",
    y="Estimated Unemployment Rate (%)",
    color="Region",
    title="Unemployment During COVID-19"
)
st.plotly_chart(fig_covid, use_container_width=True)

# Data Table
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df)

# Insights Section
st.subheader("📌 Key Insights")
st.markdown("""
• Unemployment rates increased significantly during 2020 due to COVID-19.

• Some regions experienced much higher unemployment compared to others.

• Lockdown months show the largest spikes in unemployment.

• Economic recovery can be observed in later months.
""")
