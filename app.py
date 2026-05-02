import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="AI Banking Dashboard", layout="wide")

st.title("🚀 INSANE MODE BANKING AI DASHBOARD")
st.caption("Fraud Detection • Analytics • Insights")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Comprehensive_Banking_Database.csv")

df = load_data()

# ----------------------------
# CLEAN DATA
# ----------------------------
for col in df.columns:
    if "Date" in col:
        df[col] = pd.to_datetime(df[col], errors="coerce")

for col in ["Transaction Amount", "Account Balance"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.title("Filters")

city = st.sidebar.selectbox("City", ["All"] + list(df["City"].dropna().unique()))
acct = st.sidebar.selectbox("Account Type", ["All"] + list(df["Account Type"].dropna().unique()))

view = df.copy()

if city != "All":
    view = view[view["City"] == city]

if acct != "All":
    view = view[view["Account Type"] == acct]

# ----------------------------
# SAFETY CHECK
# ----------------------------
if view.empty:
    st.warning("No data for selected filters.")
    st.stop()

# ----------------------------
# ML FRAUD DETECTION
# ----------------------------
X = view[["Transaction Amount", "Account Balance"]].fillna(0)
X = StandardScaler().fit_transform(X)

model = IsolationForest(contamination=0.02, random_state=42)
view["Risk"] = model.fit_predict(X)

# ----------------------------
# METRICS
# ----------------------------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Customers", view["Customer ID"].nunique() if "Customer ID" in view else "N/A")
c2.metric("Transactions", len(view))
c3.metric("Volume", f"${view['Transaction Amount'].sum():,.0f}")
c4.metric("Avg Balance", f"${view['Account Balance'].mean():,.0f}")
c5.metric("Fraud Alerts", (view["Risk"] == -1).sum())

# ----------------------------
# CHARTS
# ----------------------------
st.subheader("📊 Analytics Dashboard")

if "Transaction Type" in view:
    st.plotly_chart(px.pie(view, names="Transaction Type", title="Transaction Mix"))

st.plotly_chart(px.histogram(view, x="Transaction Amount", nbins=40))

if "Transaction Date" in view:
    ts = view.dropna(subset=["Transaction Date"])
    ts = ts.groupby(ts["Transaction Date"].dt.date)["Transaction Amount"].sum().reset_index()
    st.plotly_chart(px.line(ts, x="Transaction Date", y="Transaction Amount"))

# ----------------------------
# FRAUD TABLE
# ----------------------------
st.subheader("🚨 Fraud Detection Table")

fraud = view[view["Risk"] == -1]

if fraud.empty:
    st.info("No fraud detected.")
else:
    st.dataframe(fraud.head(100))

# ----------------------------
# FOOTER
# ----------------------------
st.caption("AI Banking Dashboard • Machine Learning Project")
