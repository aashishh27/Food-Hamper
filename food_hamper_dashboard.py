# food_hamper_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Food Hamper Dashboard", layout="wide")

st.title("📦 Food Hamper Analytics Dashboard")
st.markdown("Analyze trends, distributions, and neighborhood-wise demand for food hampers.")

# Load data
@st.cache_data
def load_data():
    clients_df = pd.read_csv('Clients Data Dimension(Clients_IFSSA).csv', low_memory=False)
    food_hampers_df = pd.read_excel('Food Hampers Fact.xlsx', sheet_name='FoodHampers_IFSSA')

    # Clean
    food_hampers_df['Creation Date'] = pd.to_datetime(food_hampers_df['Creation Date'], errors='coerce')
    food_hampers_df['collect_scheduled_date'] = pd.to_datetime(food_hampers_df['collect_scheduled_date'], errors='coerce')
    food_hampers_df = food_hampers_df.dropna(subset=['collect_scheduled_date'])
    food_hampers_df = food_hampers_df[food_hampers_df['appointment_type'] == 'Food Hamper']
    food_hampers_df = food_hampers_df.drop_duplicates()
    food_hampers_df.rename(columns={'client_list': 'client_id'}, inplace=True)

    # Merge
    merged_df = pd.merge(food_hampers_df, clients_df, how='left', left_on='client_id', right_on='unique id')

    # Feature engineering
    merged_df['Month'] = merged_df['collect_scheduled_date'].dt.month
    merged_df['Year'] = merged_df['collect_scheduled_date'].dt.year
    merged_df['YearMonth'] = merged_df['collect_scheduled_date'].dt.to_period('M')
    merged_df['Week'] = merged_df['collect_scheduled_date'].dt.isocalendar().week

    return merged_df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("🔎 Filters")
    years = sorted(df['Year'].dropna().unique())
    year_selected = st.multiselect("Select Year(s)", options=years, default=years)

    if 'zz_address_txt' in df.columns:
        locations = df['zz_address_txt'].dropna().unique()
        selected_location = st.selectbox("Filter by Neighborhood", options=np.append(["All"], sorted(locations)))
    else:
        selected_location = "All"

# Filter dataset
filtered_df = df[df['Year'].isin(year_selected)]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df['zz_address_txt'] == selected_location]

# KPIs
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Hampers Given", len(filtered_df))
col2.metric("Unique Clients", filtered_df['client_id'].nunique())
col3.metric("Data Range", f"{filtered_df['collect_scheduled_date'].min().date()} to {filtered_df['collect_scheduled_date'].max().date()}")

# Monthly trend chart
st.subheader("📈 Monthly Food Hamper Trend")
monthly_trend = filtered_df.groupby('YearMonth').size()
fig1, ax1 = plt.subplots(figsize=(10, 4))
monthly_trend.plot(kind='line', marker='o', ax=ax1)
ax1.set_title("Monthly Food Hamper Requests")
ax1.set_ylabel("Number of Hampers")
ax1.grid(True)
st.pyplot(fig1)

# Top 10 neighborhoods chart
if 'zz_address_txt' in df.columns:
    st.subheader("🏘️ Top 10 Neighborhoods")
    top_neighborhoods = df['zz_address_txt'].value_counts().nlargest(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_neighborhoods.values, y=top_neighborhoods.index, ax=ax2)
    ax2.set_title("Top 10 Neighborhoods by Requests")
    ax2.set_xlabel("Total Requests")
    st.pyplot(fig2)

# Heatmap
st.subheader("🔍 Correlation Heatmap (Numeric Features)")
numeric_cols = filtered_df.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
else:
    st.warning("No numeric columns available for heatmap.")

# Optional: Download cleaned data
st.download_button(
    "📥 Download Cleaned Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_food_hamper_data.csv',
    mime='text/csv'
)
