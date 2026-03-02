import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression
import datetime as dt

st.set_page_config(layout="wide")
st.title("🛒 Customer Purchase Analysis & Market Basket System")

# ---------- Upload ----------
file = st.file_uploader("Upload Online Retail Dataset", type=["xlsx"])

if file:

    df = pd.read_excel(file)

    # ---------- Cleaning ----------
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # ============================================================
    # ================== SECTION 1 ===============================
    # ============================================================

    st.header("📊 Customer Purchase Analysis")

    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # CLV
    rfm["CLV"] = rfm["Monetary"] * rfm["Frequency"]

    # Segmentation
    rfm["Segment"] = "Regular"
    rfm.loc[(rfm["Recency"] <= 30) & (rfm["Frequency"] >= 5), "Segment"] = "Loyal"
    rfm.loc[(rfm["Recency"] > 90), "Segment"] = "At Risk"

    # ---------- KPI Cards ----------
    def kpi(title, value, color):
        st.markdown(f"""
            <div style="background-color:{color};padding:20px;border-radius:10px">
            <h4>{title}</h4>
            <h2>{value}</h2>
            </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(kpi("Total Customers", rfm.shape[0], "#E3F2FD"), unsafe_allow_html=True)
    col2.markdown(kpi("Total Revenue", round(rfm["Monetary"].sum(),2), "#E8F5E9"), unsafe_allow_html=True)
    col3.markdown(kpi("Avg Frequency", round(rfm["Frequency"].mean(),2), "#FFF3E0"), unsafe_allow_html=True)
    col4.markdown(kpi("Total CLV", round(rfm["CLV"].sum(),2), "#F3E5F5"), unsafe_allow_html=True)

    # ---------- Segment Pie ----------
    st.subheader("Customer Segments")
    fig1, ax1 = plt.subplots()
    rfm["Segment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax1)
    st.pyplot(fig1)

    # ---------- Revenue by Segment ----------
    st.subheader("Revenue by Segment")
    fig2, ax2 = plt.subplots()
    rfm.groupby("Segment")["Monetary"].sum().plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

    # ---------- Recency vs Monetary ----------
    st.subheader("Recency vs Monetary")
    fig3, ax3 = plt.subplots()
    ax3.scatter(rfm["Recency"], rfm["Monetary"])
    ax3.set_xlabel("Recency")
    ax3.set_ylabel("Monetary")
    st.pyplot(fig3)

    # ---------- Top 10 CLV ----------
    st.subheader("Top 10 Customers by CLV")
    top_clv = rfm.sort_values("CLV", ascending=False).head(10)
    st.bar_chart(top_clv["CLV"])

    # ---------- Risk Indicator ----------
    risk_count = rfm[rfm["Segment"]=="At Risk"].shape[0]
    st.warning(f"⚠️ {risk_count} customers are at risk!")

    # ---------- Time Trend ----------
    st.subheader("Revenue Time Trend")
    trend = df.groupby(df["InvoiceDate"].dt.date)["TotalAmount"].sum()
    st.line_chart(trend)

    # ---------- Profit Prediction ----------
    st.subheader("Simple Revenue Prediction")

    df["DayNumber"] = (df["InvoiceDate"] - df["InvoiceDate"].min()).dt.days
    daily = df.groupby("DayNumber")["TotalAmount"].sum().reset_index()

    X = daily[["DayNumber"]]
    y = daily["TotalAmount"]

    model = LinearRegression()
    model.fit(X, y)

    future_day = st.slider("Select Future Day for Prediction", 1, 400, 365)
    prediction = model.predict([[future_day]])[0]

    st.success(f"Predicted Revenue: {round(prediction,2)}")

    # ============================================================
    # ================== SECTION 2 ===============================
    # ============================================================

    st.header("🛍️ Market Basket Analysis")

    basket = (df.groupby(["InvoiceNo", "Description"])["Quantity"]
              .sum().unstack().fillna(0))

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    min_support = st.slider("Select Minimum Support", 0.01, 0.1, 0.02)

    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=1)

    st.subheader("All Rules")
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])

    # ---------- Top 10 Lift ----------
    st.subheader("Top 10 Strongest Rules")
    top_rules = rules.sort_values("lift", ascending=False).head(10)

    fig4, ax4 = plt.subplots()
    ax4.barh(range(len(top_rules)), top_rules["lift"])
    ax4.set_yticks(range(len(top_rules)))
    ax4.set_yticklabels(top_rules["antecedents"].astype(str))
    st.pyplot(fig4)

    # ---------- Confidence vs Lift ----------
    st.subheader("Confidence vs Lift")
    fig5, ax5 = plt.subplots()
    ax5.scatter(rules["confidence"], rules["lift"])
    ax5.set_xlabel("Confidence")
    ax5.set_ylabel("Lift")
    st.pyplot(fig5)

    # ---------- Recommendation ----------
    st.subheader("Product Recommendation")

    products = list(set(df["Description"]))
    selected = st.selectbox("Select Product", products)

    rec = rules[rules["antecedents"].apply(lambda x: selected in x)]

    if not rec.empty:
        best = rec.sort_values("lift", ascending=False).iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi("Support", round(best["support"],3), "#BBDEFB"), unsafe_allow_html=True)
        c2.markdown(kpi("Confidence", round(best["confidence"],3), "#C8E6C9"), unsafe_allow_html=True)
        c3.markdown(kpi("Lift", round(best["lift"],3), "#FFE0B2"), unsafe_allow_html=True)

        st.success(f"Recommended Product: {list(best['consequents'])}")
    else:
        st.info("No strong recommendation found.")
