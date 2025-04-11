import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import joblib
import os
import datetime as dt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# For RAG Chatbot
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader

# Set Streamlit page configuration
st.set_page_config(page_title="Food Hamper Dashboard", layout="wide")

st.title("📦 Food Hamper Analytics & AI Dashboard")
st.markdown("Explore data trends, predict demand, and interact with AI-driven insights.")

# Load data
@st.cache_data
def load_data():
    clients_df = pd.read_csv('Clients Data Dimension(Clients_IFSSA).csv', low_memory=False)
    food_hampers_df = pd.read_excel('Food Hampers Fact.xlsx', sheet_name='FoodHampers_IFSSA')
    
    food_hampers_df['Creation Date'] = pd.to_datetime(food_hampers_df['Creation Date'], errors='coerce')
    food_hampers_df['collect_scheduled_date'] = pd.to_datetime(food_hampers_df['collect_scheduled_date'], errors='coerce')
    food_hampers_df = food_hampers_df.dropna(subset=['collect_scheduled_date'])
    food_hampers_df = food_hampers_df[food_hampers_df['appointment_type'] == 'Food Hamper']
    food_hampers_df = food_hampers_df.drop_duplicates()
    food_hampers_df.rename(columns={'client_list': 'client_id'}, inplace=True)

    merged_df = pd.merge(food_hampers_df, clients_df, how='left', left_on='client_id', right_on='unique id')

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

filtered_df = df[df['Year'].isin(year_selected)]
if selected_location != "All":
    filtered_df = filtered_df[filtered_df['zz_address_txt'] == selected_location]

# Tabs Layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA & Trends", "🧼 Cleaning & Features", "🧠 ML Modeling", 
    "💡 Explainable AI", "⏳ Time Series", "🤖 RAG Chatbot"
])

# Tab 1: EDA
with tab1:
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Hampers Given", len(filtered_df))
    col2.metric("Unique Clients", filtered_df['client_id'].nunique())
    col3.metric("Date Range", f"{filtered_df['collect_scheduled_date'].min().date()} to {filtered_df['collect_scheduled_date'].max().date()}")

    st.subheader("📈 Monthly Trend")
    monthly = filtered_df.groupby('YearMonth').size()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind='line', marker='o', ax=ax)
    ax.set_ylabel("No. of Hampers")
    st.pyplot(fig)

    if 'zz_address_txt' in filtered_df.columns:
        st.subheader("🏘️ Top Neighborhoods")
        top_neigh = filtered_df['zz_address_txt'].value_counts().nlargest(10)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=top_neigh.values, y=top_neigh.index, ax=ax2)
        st.pyplot(fig2)

# Tab 2: Cleaning & Feature Engineering
with tab2:
    st.subheader("🧼 Data Cleaning & Feature Engineering")
    st.markdown("""
    - Removed null `collect_scheduled_date`
    - Filtered only `Food Hamper` appointments
    - Created features: `Month`, `Year`, `YearMonth`, `Week`
    """)

    st.write("Sample cleaned dataset:")
    st.dataframe(filtered_df.head())

# Tab 3: ML Modeling and Optimization
with tab3:
    st.subheader("🧠 ML: Predict Repeat Clients")
    
    model_df = filtered_df.copy()
    model_df['RepeatClient'] = model_df.duplicated(subset='client_id', keep=False).astype(int)

    le = LabelEncoder()
    model_df['Neighborhood'] = le.fit_transform(model_df['zz_address_txt'].astype(str))

    features = ['Month', 'Year', 'Week', 'Neighborhood']
    X = model_df[features]
    y = model_df['RepeatClient']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Hyperparameter Optimization
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    y_pred = best_clf.predict(X_test)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

# Tab 4: Explainable AI (SHAP)
with tab4:
    st.subheader("💡 SHAP Explanation of Predictions")

    explainer = shap.TreeExplainer(best_clf)
    shap_values = explainer.shap_values(X)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("### Global Feature Importance")
    shap.summary_plot(shap_values[1], X, plot_type="bar", show=False)
    st.pyplot()

    st.markdown("### Detailed SHAP Summary")
    shap.summary_plot(shap_values[1], X, show=False)
    st.pyplot()

# Tab 5: Time Series Analysis
with tab5:
    st.subheader("📅 Weekly Trend Analysis")

    weekly_trend = filtered_df.groupby(['Year', 'Week']).size().reset_index(name='count')
    weekly_trend['Date'] = pd.to_datetime(weekly_trend['Year'].astype(str) + '-W' + weekly_trend['Week'].astype(str) + '-1', format='%G-W%V-%u')

    fig4 = px.line(weekly_trend, x='Date', y='count', title="Weekly Distribution of Hampers")
    st.plotly_chart(fig4)

# Tab 6: RAG Chatbot
with tab6:
    st.subheader("🤖 Retrieval-Augmented Generation Chatbot")

    st.markdown("""
    Upload your CSV and ask questions about it. The AI will retrieve context from your data to provide accurate answers.
    """)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        file_path = f"uploaded_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = CSVLoader(file_path)
        documents = loader.load()

        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(documents, embeddings)
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), chain_type="stuff", retriever=retriever)

        question = st.text_input("Ask a question about your data:")
        if question:
            with st.spinner("Thinking..."):
                answer = qa.run(question)
                st.success(answer)
