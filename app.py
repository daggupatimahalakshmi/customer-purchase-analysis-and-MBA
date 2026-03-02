import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Retail Analytics System",
                   layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: bold;
}
.kpi1 {color:#FF4B4B;}
.kpi2 {color:#1F77B4;}
.kpi3 {color:#2CA02C;}
</style>
""", unsafe_allow_html=True)

st.title("🛒 Retail Analytics Dashboard")

# ---------------- LOAD FILES ----------------
rfm = pickle.load(open("rfm.pkl","rb"))
rules = pickle.load(open("rules.pkl","rb"))

rfm = rfm.copy()

# Fix index issue
if rfm.index.name == "CustomerID":
    rfm = rfm.reset_index()

# ---------------- CREATE SEGMENTS ----------------
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1], duplicates='drop')
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4], duplicates='drop')
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4], duplicates='drop')

rfm['RFM_Score'] = (
    rfm['R_Score'].astype(str) +
    rfm['F_Score'].astype(str) +
    rfm['M_Score'].astype(str)
)

def segment(x):
    if x >= "444":
        return "Champions"
    elif x >= "344":
        return "Loyal"
    elif x >= "244":
        return "Potential"
    else:
        return "At Risk"

rfm["Segment"] = rfm["RFM_Score"].apply(segment)

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio("Select Section",
                        ["Customer Purchase Analysis",
                         "Market Basket Analysis"])

# ============================================================
# 1️⃣ CUSTOMER PURCHASE ANALYSIS
# ============================================================
if menu == "Customer Purchase Analysis":

    st.header("📊 Customer Purchase Analysis")

    # ---------------- KPI CARDS ----------------
    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<h3 class='kpi1'>Total Customers</h3>", unsafe_allow_html=True)
    col1.metric("", rfm.shape[0])

    col2.markdown(f"<h3 class='kpi2'>Average Spend</h3>", unsafe_allow_html=True)
    col2.metric("", round(rfm["Monetary"].mean(),2))

    col3.markdown(f"<h3 class='kpi3'>High Value Customers</h3>", unsafe_allow_html=True)
    col3.metric("", rfm[rfm["Segment"]=="Champions"].shape[0])

    # ---------------- PIE CHART ----------------
    st.subheader("Customer Segment Distribution")

    fig1, ax1 = plt.subplots()
    rfm["Segment"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
    st.pyplot(fig1)

    # ---------------- SLIDER FILTER ----------------
    st.subheader("Filter by Minimum Monetary Value")

    min_value = st.slider("Select Minimum Spend",
                          int(rfm["Monetary"].min()),
                          int(rfm["Monetary"].max()),
                          100)

    filtered = rfm[rfm["Monetary"] >= min_value]
    st.dataframe(filtered.head(20))

    # ---------------- BAR CHART ----------------
    st.subheader("Segment Wise Average Spend")

    seg_avg = rfm.groupby("Segment")["Monetary"].mean()
    st.bar_chart(seg_avg)

# ============================================================
# 2️⃣ MARKET BASKET ANALYSIS
# ============================================================
elif menu == "Market Basket Analysis":

    st.header("🛍 Market Basket Analysis")

    # ---------------- SLIDERS ----------------
    min_support = st.slider("Minimum Support", 0.01, 0.5, 0.02)
    min_conf = st.slider("Minimum Confidence", 0.1, 1.0, 0.3)
    min_lift = st.slider("Minimum Lift", 0.5, 5.0, 1.0)

    filtered_rules = rules[
        (rules["support"] >= min_support) &
        (rules["confidence"] >= min_conf) &
        (rules["lift"] >= min_lift)
    ]

    st.subheader("All Association Rules")
    st.dataframe(filtered_rules[[
        "antecedents","consequents",
        "support","confidence","lift"
    ]])

    # ---------------- VISUALIZATION ----------------
    st.subheader("Support vs Confidence Chart")

    fig2, ax2 = plt.subplots()
    ax2.scatter(filtered_rules["support"],
                filtered_rules["confidence"])
    ax2.set_xlabel("Support")
    ax2.set_ylabel("Confidence")
    st.pyplot(fig2)

    # ---------------- RECOMMENDATION ----------------
    st.subheader("🎯 Product Recommendation")

    products = list(set(
        item for sublist in rules["antecedents"]
        for item in sublist
    ))

    selected = st.selectbox("Select Product", products)

    rec = rules[rules["antecedents"].apply(lambda x: selected in x)]

    if not rec.empty:

        best = rec.sort_values("lift", ascending=False).iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.markdown("<h4 style='color:#FF4B4B'>Support</h4>", unsafe_allow_html=True)
        col1.metric("", round(best["support"],3))

        col2.markdown("<h4 style='color:#1F77B4'>Confidence</h4>", unsafe_allow_html=True)
        col2.metric("", round(best["confidence"],3))

        col3.markdown("<h4 style='color:#2CA02C'>Lift</h4>", unsafe_allow_html=True)
        col3.metric("", round(best["lift"],3))

        st.success(f"Recommended Product: {list(best['consequents'])}")

        # Lift Visualization
        fig3, ax3 = plt.subplots()
        ax3.bar(["Support","Confidence","Lift"],
                [best["support"], best["confidence"], best["lift"]])
        st.pyplot(fig3)

    else:
        st.warning("No strong recommendation found.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("👩‍💻 Developed by Maha | Retail Analytics Project")
