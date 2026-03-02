import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression
import datetime as dt

st.set_page_config(layout="wide")
st.title("🛒 Retail Analytics Dashboard")

# ==============================
# Load Dataset Automatically
# ==============================

df = pd.read_excel("Online Retail.xlsx")

df = df.dropna(subset=["CustomerID"])
df = df[df["Quantity"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

# ==============================
# Sidebar Navigation
# ==============================

section = st.sidebar.radio(
    "Select Analysis Section",
    ["Customer Purchase Analysis", "Market Basket Analysis"]
)

# ==============================
# KPI Function
# ==============================

def kpi(title, value, color):
    return f"""
        <div style="background-color:{color};
                    padding:20px;
                    border-radius:10px;
                    text-align:center">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
    """

# ==========================================================
# ================= CUSTOMER PURCHASE =======================
# ==========================================================

if section == "Customer Purchase Analysis":

    st.header("📊 Customer Purchase Analysis")

    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    rfm["CLV"] = rfm["Monetary"] * rfm["Frequency"]

    rfm["Segment"] = "Regular"
    rfm.loc[(rfm["Recency"] <= 30) & (rfm["Frequency"] >= 5), "Segment"] = "Loyal"
    rfm.loc[(rfm["Recency"] > 90), "Segment"] = "At Risk"

    # -------- Sliders --------
    recency_filter = st.slider(
        "Max Recency Filter",
        int(rfm["Recency"].min()),
        int(rfm["Recency"].max()),
        100
    )

    freq_filter = st.slider(
        "Minimum Frequency",
        int(rfm["Frequency"].min()),
        int(rfm["Frequency"].max()),
        1
    )

    filtered_rfm = rfm[
        (rfm["Recency"] <= recency_filter) &
        (rfm["Frequency"] >= freq_filter)
    ]

    # -------- KPIs --------
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(kpi("Total Customers",
                      filtered_rfm.shape[0],
                      "#E3F2FD"), unsafe_allow_html=True)

    col2.markdown(kpi("Total Revenue",
                      round(filtered_rfm["Monetary"].sum(),2),
                      "#E8F5E9"), unsafe_allow_html=True)

    col3.markdown(kpi("Avg Frequency",
                      round(filtered_rfm["Frequency"].mean(),2),
                      "#FFF3E0"), unsafe_allow_html=True)

    col4.markdown(kpi("Total CLV",
                      round(filtered_rfm["CLV"].sum(),2),
                      "#F3E5F5"), unsafe_allow_html=True)

    # -------- Segment Pie --------
    st.subheader("Customer Segments")
    fig1, ax1 = plt.subplots()
    filtered_rfm["Segment"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=ax1)
    st.pyplot(fig1)

    # -------- Revenue by Segment --------
    st.subheader("Revenue by Segment")
    fig2, ax2 = plt.subplots()
    filtered_rfm.groupby("Segment")["Monetary"].sum().plot(
        kind="bar", ax=ax2)
    st.pyplot(fig2)

    # -------- Recency vs Monetary --------
    st.subheader("Recency vs Monetary")
    fig3, ax3 = plt.subplots()
    ax3.scatter(filtered_rfm["Recency"],
                filtered_rfm["Monetary"])
    ax3.set_xlabel("Recency")
    ax3.set_ylabel("Monetary")
    st.pyplot(fig3)

    # -------- Time Trend --------
    st.subheader("Revenue Time Trend")
    trend = df.groupby(
        df["InvoiceDate"].dt.date)["TotalAmount"].sum()
    st.line_chart(trend)

    # -------- Revenue Prediction --------
    st.subheader("Revenue Prediction")

    df["DayNumber"] = (
        df["InvoiceDate"] - df["InvoiceDate"].min()
    ).dt.days

    daily = df.groupby("DayNumber")[
        "TotalAmount"].sum().reset_index()

    model = LinearRegression()
    model.fit(daily[["DayNumber"]],
              daily["TotalAmount"])

    future_day = st.slider(
        "Select Future Day",
        1, 400, 365
    )

    prediction = model.predict([[future_day]])[0]

    st.success(f"Predicted Revenue: {round(prediction,2)}")

# ==========================================================
# ================= MARKET BASKET ==========================
# ==========================================================

elif section == "Market Basket Analysis":

    st.header("🛍️ Market Basket Analysis")

    basket = (df.groupby(["InvoiceNo", "Description"])["Quantity"]
              .sum().unstack().fillna(0))

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    min_support = st.slider(
        "Minimum Support",
        0.01, 0.1, 0.02
    )

    min_conf = st.slider(
        "Minimum Confidence",
        0.1, 1.0, 0.5
    )

    frequent = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=min_conf
    )

    st.subheader("Association Rules")
    st.dataframe(
        rules[["antecedents",
               "consequents",
               "support",
               "confidence",
               "lift"]]
    )

    # -------- Lift Scatter --------
    st.subheader("Confidence vs Lift")
    fig4, ax4 = plt.subplots()
    ax4.scatter(rules["confidence"],
                rules["lift"])
    ax4.set_xlabel("Confidence")
    ax4.set_ylabel("Lift")
    st.pyplot(fig4)

    # -------- Recommendation --------
    st.subheader("Product Recommendation")

    products = list(df["Description"].unique())
    selected = st.selectbox(
        "Select Product",
        products
    )

    rec = rules[
        rules["antecedents"].apply(
            lambda x: selected in x)
    ]

    if not rec.empty:
        best = rec.sort_values(
            "lift",
            ascending=False
        ).iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.markdown(
            kpi("Support",
                round(best["support"],3),
                "#BBDEFB"),
            unsafe_allow_html=True)

        c2.markdown(
            kpi("Confidence",
                round(best["confidence"],3),
                "#C8E6C9"),
            unsafe_allow_html=True)

        c3.markdown(
            kpi("Lift",
                round(best["lift"],3),
                "#FFE0B2"),
            unsafe_allow_html=True)

        st.success(
            f"Recommended Product: {list(best['consequents'])}"
        )
    else:
        st.info("No strong recommendation found.")
