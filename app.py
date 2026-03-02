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
</style>
""", unsafe_allow_html=True)

st.title("🛒 Retail Analytics Dashboard")

# ---------------- LOAD PICKLE FILES ----------------
with open("rfm.pkl", "rb") as f:
    rfm = pickle.load(f)

with open("rules.pkl", "rb") as f:
    rules = pickle.load(f)

rfm = rfm.copy()

if rfm.index.name == "CustomerID":
    rfm = rfm.reset_index()

# ---------------- CREATE RFM SEGMENTS ----------------
rfm['R_Score'] = pd.qcut(rfm['Recency'], 4,
                         labels=[4,3,2,1], duplicates='drop')

rfm['F_Score'] = pd.qcut(
    rfm['Frequency'].rank(method='first'),
    4, labels=[1,2,3,4], duplicates='drop')

rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4,
                         labels=[1,2,3,4], duplicates='drop')

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
menu = st.sidebar.radio(
    "📂 Navigation",
    ["🏠 Home",
     "📊 Customer Purchase Analysis",
     "🛍 Market Basket Analysis"]
)

# ============================================================
# 🏠 HOME – BUSINESS INSIGHTS
# ============================================================
if menu == "🏠 Home":

    st.header("📊 Executive Business Insights")

    total_customers = rfm.shape[0]
    total_revenue = round(rfm["Monetary"].sum(), 2)
    avg_spend = round(rfm["Monetary"].mean(), 2)
    champions = rfm[rfm["Segment"]=="Champions"].shape[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", total_customers)
    col2.metric("Total Revenue", total_revenue)
    col3.metric("Average Spend", avg_spend)
    col4.metric("Champions Customers", champions)

    st.markdown("---")

    st.subheader("📌 Customer Segment Distribution")
    st.bar_chart(rfm["Segment"].value_counts())

    st.markdown("---")

    st.subheader("🛍 Strongest Product Association")

    if not rules.empty:
        top_rule = rules.sort_values("lift",
                                     ascending=False).iloc[0]

        st.success(f"""
        If customers buy {list(top_rule['antecedents'])}
        they are highly likely to buy {list(top_rule['consequents'])}

        Lift: {round(top_rule['lift'],2)}
        Confidence: {round(top_rule['confidence'],2)}
        """)

    st.markdown("---")

    st.subheader("🚀 Strategic Recommendations")
    st.markdown("""
    ✔ Focus marketing on Champions segment  
    ✔ Retarget At Risk customers with offers  
    ✔ Bundle high-lift products together  
    ✔ Use recommendations for cross-selling  
    ✔ Improve loyalty rewards for repeat buyers  
    """)

# ============================================================
# 📊 CUSTOMER PURCHASE ANALYSIS
# ============================================================
elif menu == "📊 Customer Purchase Analysis":

    st.header("📊 Customer Purchase Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", rfm.shape[0])
    col2.metric("Average Spend",
                round(rfm["Monetary"].mean(),2))
    col3.metric("High Value Customers",
                rfm[rfm["Segment"]=="Champions"].shape[0])

    st.subheader("Customer Segment Distribution")

    fig1, ax1 = plt.subplots()
    rfm["Segment"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Filter by Minimum Monetary Value")

    min_value = st.slider(
        "Select Minimum Spend",
        int(rfm["Monetary"].min()),
        int(rfm["Monetary"].max()),
        100
    )

    filtered = rfm[rfm["Monetary"] >= min_value]
    st.dataframe(filtered.head(20))

    st.subheader("Segment Wise Average Spend")
    st.bar_chart(
        rfm.groupby("Segment")["Monetary"].mean()
    )

# ============================================================
# 🛍 MARKET BASKET ANALYSIS
# ============================================================
elif menu == "🛍 Market Basket Analysis":

    st.header("🛍 Market Basket Analysis")

    min_support = st.slider(
        "Minimum Support", 0.01, 0.5, 0.02)

    min_conf = st.slider(
        "Minimum Confidence", 0.1, 1.0, 0.3)

    min_lift = st.slider(
        "Minimum Lift", 0.5, 5.0, 1.0)

    filtered_rules = rules[
        (rules["support"] >= min_support) &
        (rules["confidence"] >= min_conf) &
        (rules["lift"] >= min_lift)
    ]

    st.subheader("All Association Rules")
    st.dataframe(filtered_rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]])

    st.subheader("Support vs Confidence")

    fig2, ax2 = plt.subplots()
    ax2.scatter(filtered_rules["support"],
                filtered_rules["confidence"])
    ax2.set_xlabel("Support")
    ax2.set_ylabel("Confidence")
    st.pyplot(fig2)

    st.subheader("🎯 Product Recommendation")

    products = list(set(
        item for sublist in rules["antecedents"]
        for item in sublist
    ))

    selected = st.selectbox("Select Product", products)

    rec = rules[
        rules["antecedents"].apply(
            lambda x: selected in x)
    ]

    if not rec.empty:

        best = rec.sort_values(
            "lift",
            ascending=False).iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Support",
                    round(best["support"],3))
        col2.metric("Confidence",
                    round(best["confidence"],3))
        col3.metric("Lift",
                    round(best["lift"],3))

        st.success(
            f"Recommended Product: {list(best['consequents'])}"
        )

        fig3, ax3 = plt.subplots()
        ax3.bar(["Support","Confidence","Lift"],
                [best["support"],
                 best["confidence"],
                 best["lift"]])
        st.pyplot(fig3)

    else:
        st.warning("No strong recommendation found.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("👩‍💻 Developed by Maha | Retail Analytics Project")
