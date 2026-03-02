import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Purchase Analysis", layout="wide")

st.title("🛒 Customer Purchase Pattern & Basket Analysis")

# Load pickle files
with open("rfm.pkl", "rb") as f:
    rfm = pickle.load(f)

with open("rules.pkl", "rb") as f:
    rules = pickle.load(f)

menu = st.sidebar.selectbox("Select Analysis",
                            ["Dashboard",
                             "RFM Analysis",
                             "Market Basket",
                             "Recommendation"])

# ---------------- Dashboard ----------------
if menu == "Dashboard":
    st.subheader("Business Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", rfm.shape[0])
    col2.metric("Total Rules", rules.shape[0])
    col3.metric("Average Monetary Value", round(rfm["Monetary"].mean(),2))

# ---------------- RFM ----------------
elif menu == "RFM Analysis":
    st.subheader("Top 10 Customers by Monetary Value")
    top = rfm.sort_values("Monetary", ascending=False).head(10)
    st.dataframe(top)

    st.subheader("Monetary Distribution")
    fig, ax = plt.subplots()
    ax.hist(rfm["Monetary"], bins=30)
    st.pyplot(fig)

# ---------------- Market Basket ----------------
elif menu == "Market Basket":
    st.subheader("Top Association Rules")
    st.dataframe(rules[['antecedents','consequents',
                        'support','confidence','lift']].head(10))

# ---------------- Recommendation ----------------
elif menu == "Recommendation":
    st.subheader("Product Recommendation")

    products = list(set([item for sublist in rules['antecedents'] for item in sublist]))
    selected = st.selectbox("Select Product", products)

    rec = rules[rules['antecedents'].apply(lambda x: selected in x)]

    if not rec.empty:
        st.write("Recommended Products:")
        st.dataframe(rec[['consequents','confidence','lift']]
                     .sort_values(by='lift', ascending=False)
                     .head(5))
    else:
        st.write("No recommendation found.")