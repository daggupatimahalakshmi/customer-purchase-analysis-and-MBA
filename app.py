import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Purchase Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS (Power BI Style) ----------------
st.markdown("""
<style>
.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
    text-align: center;
}
.metric-title {
    font-size: 18px;
    color: #555;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #6C63FF;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Purchase Pattern & Basket Analysis System")
st.markdown("---")

# ---------------- LOAD DATA ----------------
with open("rfm.pkl", "rb") as f:
    rfm = pickle.load(f)

with open("rules.pkl", "rb") as f:
    rules = pickle.load(f)

# ---------------- CREATE SEGMENTS ----------------
rfm['Segment'] = pd.qcut(rfm['Monetary'], 
                         q=4, 
                         labels=["Low Value", "Mid Value", "High Value", "Premium"])

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio("📌 Navigation",
                        ["📊 Dashboard",
                         "👥 Customer Segmentation",
                         "🛍 Market Basket Analysis",
                         "🎯 Recommendation Engine"])

# ==================================================
# 📊 DASHBOARD
# ==================================================
if menu == "📊 Dashboard":

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Customers</div>
        <div class="metric-value">{rfm.shape[0]}</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Rules</div>
        <div class="metric-value">{rules.shape[0]}</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Spend</div>
        <div class="metric-value">₹ {round(rfm['Monetary'].mean(),2)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📌 Key Insights")
    st.info("""
    - Top customers contribute significant revenue.
    - RFM used to segment customers by spending behavior.
    - Association rules generated using Apriori algorithm.
    - Recommendation engine built using Lift metric.
    """)

# ==================================================
# 👥 CUSTOMER SEGMENTATION
# ==================================================
elif menu == "👥 Customer Segmentation":

    st.subheader("Customer Segment Distribution")

    segment_counts = rfm['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']

    fig = px.pie(segment_counts,
                 names='Segment',
                 values='Count',
                 title="Customer Segments",
                 color_discrete_sequence=px.colors.sequential.Purples)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Premium Customers")
    premium = rfm[rfm['Segment'] == "Premium"]\
        .sort_values("Monetary", ascending=False)\
        .head(10)

    st.dataframe(premium)

# ==================================================
# 🛍 MARKET BASKET ANALYSIS
# ==================================================
elif menu == "🛍 Market Basket Analysis":

    st.subheader("Association Rules")

    min_lift = st.slider("Select Minimum Lift", 
                         float(rules['lift'].min()), 
                         float(rules['lift'].max()), 
                         1.0)

    filtered = rules[rules['lift'] >= min_lift]

    st.dataframe(filtered[['antecedents',
                           'consequents',
                           'support',
                           'confidence',
                           'lift']].sort_values(by="lift", ascending=False).head(10))

    st.subheader("Lift vs Confidence")

    fig = px.scatter(filtered,
                     x="lift",
                     y="confidence",
                     size="support",
                     title="Lift vs Confidence",
                     color="lift",
                     color_continuous_scale="purples")

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 🎯 RECOMMENDATION ENGINE
# ==================================================
elif menu == "🎯 Recommendation Engine":

    st.subheader("Product Recommendation System")

    products = list(set([item for sublist in rules['antecedents'] for item in sublist]))
    selected = st.selectbox("Select Product", products)

    rec = rules[rules['antecedents'].apply(lambda x: selected in x)]

    if not rec.empty:
        st.success("Recommended Products:")
        st.dataframe(rec[['consequents',
                          'confidence',
                          'lift']]
                     .sort_values(by='lift', ascending=False)
                     .head(5))
    else:
        st.warning("No strong recommendation found.")
