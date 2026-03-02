import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- POWER BI STYLE ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #F4F6F9;
}
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: bold;
    color: #6C63FF;
}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Customer Purchase Pattern & Basket Analysis System")

# ---------------- LOAD DATA ----------------
with open("rfm.pkl", "rb") as f:
    rfm = pickle.load(f)

with open("rules.pkl", "rb") as f:
    rules = pickle.load(f)

# If original dataset available for trend
try:
    df = pd.read_pickle("retail.pkl")
except:
    df = None

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")

menu = st.sidebar.radio("Go to",
                        ["📊 Dashboard",
                         "👥 Segmentation",
                         "🔍 Customer Lookup",
                         "📈 Sales Trend",
                         "💰 Profit Prediction",
                         "🎯 Recommendation"])

st.sidebar.markdown("## 📘 Project Description")
st.sidebar.info("""
• RFM Customer Segmentation  
• Market Basket Analysis (Apriori)  
• Association Rule Mining  
• Product Recommendation Engine  
• Sales Trend Analytics  
• Machine Learning Profit Prediction  

Built using Python & Streamlit
""")

# ---------------- DASHBOARD ----------------
if menu == "📊 Dashboard":

    st.subheader("📈 Business Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Customers", rfm.shape[0])
    col2.metric("📦 Total Rules", rules.shape[0])
    col3.metric("💰 Avg Spend", round(rfm["Monetary"].mean(),2))

    st.markdown("### 🏷 Customer Segment Distribution")

    segment_count = rfm["Segment"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(segment_count, labels=segment_count.index, autopct="%1.1f%%")
    st.pyplot(fig)

# ---------------- SEGMENTATION ----------------
elif menu == "👥 Segmentation":

    st.subheader("Customer Segment Analysis")

    slider_value = st.slider("Minimum Monetary Value", 
                             int(rfm["Monetary"].min()),
                             int(rfm["Monetary"].max()),
                             100)

    filtered = rfm[rfm["Monetary"] >= slider_value]

    st.dataframe(filtered.head(20))

    st.subheader("Segment Wise Average Spend")

    seg_avg = rfm.groupby("Segment")["Monetary"].mean()
    st.bar_chart(seg_avg)

# ---------------- CUSTOMER LOOKUP ----------------
elif menu == "🔍 Customer Lookup":

    st.subheader("Search Customer by ID")

    customer_id = st.text_input("Enter Customer ID")

    if customer_id:
        cust = rfm[rfm["CustomerID"].astype(str) == customer_id]

        if not cust.empty:
            col1, col2, col3 = st.columns(3)

            col1.metric("Recency", int(cust["Recency"].values[0]))
            col2.metric("Frequency", int(cust["Frequency"].values[0]))
            col3.metric("Monetary", round(cust["Monetary"].values[0],2))

            st.success(f"Segment: {cust['Segment'].values[0]}")
        else:
            st.error("Customer not found")

# ---------------- SALES TREND ----------------
elif menu == "📈 Sales Trend":

    if df is not None:

        st.subheader("Monthly Sales Trend")

        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Month'] = df['InvoiceDate'].dt.to_period('M')

        monthly_sales = df.groupby('Month')['TotalPrice'].sum()
        monthly_sales.index = monthly_sales.index.astype(str)

        st.line_chart(monthly_sales)

    else:
        st.warning("Sales dataset not found (retail.pkl missing)")

# ---------------- PROFIT PREDICTION ----------------
elif menu == "💰 Profit Prediction":

    st.subheader("Profit Prediction Model")

    X = rfm[['Recency','Frequency']]
    y = rfm['Monetary']

    model = LinearRegression()
    model.fit(X, y)

    rec = st.slider("Recency", 1, 365, 30)
    freq = st.slider("Frequency", 1, 100, 5)

    prediction = model.predict([[rec, freq]])

    st.success(f"Predicted Customer Profit: ₹ {round(prediction[0],2)}")

# ---------------- RECOMMENDATION ----------------
elif menu == "🎯 Recommendation":

    st.subheader("Product Recommendation Engine")

    products = list(set([item for sublist in rules['antecedents'] for item in sublist]))
    selected = st.selectbox("Select a Product", products)

    rec = rules[rules['antecedents'].apply(lambda x: selected in x)]

    if not rec.empty:

        best = rec.sort_values("lift", ascending=False).iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Support", round(best["support"],3))
        col2.metric("Confidence", round(best["confidence"],3))
        col3.metric("Lift", round(best["lift"],3))

        st.success(f"Recommended Product: {list(best['consequents'])}")

    else:
        st.warning("No strong recommendation found.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("👩‍💻 Developed by Maha | Retail Analytics Project")
