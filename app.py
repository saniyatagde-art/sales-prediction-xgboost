from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ---------- App configuration ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sales_xgboost_model.pkl"
DATA_PATH = BASE_DIR / "sales_prediction_dataset.csv"

st.set_page_config(
    page_title="Sales Forecast | XGBoost",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
:root { --navy:#10243a; --blue:#2563eb; --cyan:#38bdf8; --muted:#64748b; }
[data-testid="stAppViewContainer"] { background: #f4f7fb; }
[data-testid="stHeader"] { background: rgba(244,247,251,.92); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#10243a 0%,#183b5b 100%); }
[data-testid="stSidebar"] * { color: #f8fafc !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { 
    background: rgba(255,255,255,.08); border-radius: 10px; padding: 9px 12px; margin: 4px 0;
}
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px; }
.hero {
    background: linear-gradient(115deg,#10243a 0%,#1d4e89 62%,#2563eb 100%);
    padding: 30px 34px; border-radius: 20px; color: white; margin-bottom: 22px;
    box-shadow: 0 12px 30px rgba(37,99,235,.15);
}
.hero h1 { color:white; font-size:2.15rem; margin:0 0 8px 0; letter-spacing:-.5px; }
.hero p { color:#dbeafe; font-size:1rem; margin:0; }
.section-card {
    background: white; padding: 20px 22px; border:1px solid #e5edf5;
    border-radius:16px; box-shadow:0 5px 18px rgba(15,23,42,.04); margin-bottom:16px;
}
.kpi {
    background:white; border:1px solid #e5edf5; border-radius:15px; padding:17px 18px;
    box-shadow:0 5px 18px rgba(15,23,42,.04); min-height:108px;
}
.kpi-label { color:#64748b; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.kpi-value { color:#10243a; font-size:1.65rem; font-weight:750; margin-top:7px; }
.result {
    background:linear-gradient(120deg,#eaf4ff,#f0f9ff); border:1px solid #bfdbfe;
    border-radius:18px; padding:24px; text-align:center; margin-top:18px;
}
.result-label { color:#1e40af; font-weight:700; text-transform:uppercase; letter-spacing:.08em; font-size:.85rem; }
.result-value { color:#10243a; font-size:2.5rem; font-weight:800; margin:7px 0; }
.small-note { color:#64748b; font-size:.85rem; }
div.stButton > button[kind="primary"] {
    background:linear-gradient(90deg,#2563eb,#0ea5e9); color:white; border:0;
    border-radius:10px; padding:.65rem 1rem; font-weight:700;
}
div.stButton > button[kind="primary"]:hover { border:0; filter:brightness(.96); }
h1,h2,h3 { color:#10243a; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH.name}")
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    data = load_data()
except Exception as exc:
    st.error(f"Could not load the project files. Check that the model and CSV are in the same folder as app.py. Details: {exc}")
    st.stop()

required_cols = {
    "Advertising_Spend", "Product_Price", "Discount",
    "Previous_Month_Sales", "Store_Type", "Product_Category", "Sales"
}
if not required_cols.issubset(data.columns):
    st.error("The dataset columns do not match the expected project format.")
    st.stop()

# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown("## 📈 Sales Forecast")
    st.caption("XGBoost Regression Application")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Predict Sales", "Sales Analytics", "About Project"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("Academic ML project")
    st.caption("Dataset: 1,000 synthetic records")

# ---------- Shared header ----------
st.markdown("""
<div class="hero">
  <h1>Sales Prediction using XGBoost</h1>
  <p>Machine learning system for forecasting future sales performance</p>
</div>
""", unsafe_allow_html=True)

# ---------- Overview ----------
if page == "Overview":
    st.subheader("Project Overview")
    st.write("Estimate sales using advertising spend, product pricing, discounts, previous-month sales, store type and product category.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Dataset records</div><div class="kpi-value">{len(data):,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi"><div class="kpi-label">Prediction inputs</div><div class="kpi-value">6 features</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi"><div class="kpi-label">Algorithm</div><div class="kpi-value">XGBoost</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.15, .85])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### How the system works")
        st.markdown("**01**  Enter sales-related values  \n**02**  Submit the input form  \n**03**  The saved XGBoost model processes the values  \n**04**  View the estimated sales")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Model evaluation")
        st.metric("MAE", "2,527.35")
        st.metric("RMSE", "3,202.70")
        st.metric("R² Score", "0.9540")
        st.markdown("</div>", unsafe_allow_html=True)
    st.info("The dataset is synthetic and intended for academic demonstration. Predictions are estimates, not guaranteed business outcomes.")

# ---------- Prediction ----------
elif page == "Predict Sales":
    st.subheader("Enter business details")
    st.write("Fill in the six input fields, then select **Predict Sales**.")
    with st.form("sales_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            advertising_spend = st.number_input("Advertising Spend (₹)", min_value=0.0, value=50000.0, step=1000.0)
            product_price = st.number_input("Product Price (₹)", min_value=1.0, value=200.0, step=10.0)
            discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        with col2:
            previous_month_sales = st.number_input("Previous Month Sales (₹)", min_value=0.0, value=30000.0, step=1000.0)
            store_type = st.selectbox("Store Type", sorted(data["Store_Type"].dropna().unique().tolist()))
            product_category = st.selectbox("Product Category", sorted(data["Product_Category"].dropna().unique().tolist()))
        submitted = st.form_submit_button("✨ Predict Sales", type="primary", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "Advertising_Spend": advertising_spend,
            "Product_Price": product_price,
            "Discount": discount,
            "Previous_Month_Sales": previous_month_sales,
            "Store_Type": store_type,
            "Product_Category": product_category,
        }])
        try:
            prediction = float(model.predict(input_data)[0])
            st.markdown(
                f'<div class="result"><div class="result-label">Estimated Sales</div>'
                f'<div class="result-value">₹ {prediction:,.2f}</div>'
                f'<div class="small-note">Generated by the saved XGBoost regression model</div></div>',
                unsafe_allow_html=True,
            )
            st.caption("This estimate is based on the model and synthetic training data supplied with the project.")
        except Exception as exc:
            st.error(f"Prediction could not be generated. Check model compatibility and input preprocessing. Details: {exc}")

# ---------- Analytics ----------
elif page == "Sales Analytics":
    st.subheader("Dataset analytics")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Average sales by product category")
        category_sales = data.groupby("Product_Category")["Sales"].mean().sort_values(ascending=False)
        st.bar_chart(category_sales, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Average sales by store type")
        store_sales = data.groupby("Store_Type")["Sales"].mean().sort_values(ascending=False)
        st.bar_chart(store_sales, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Advertising spend and sales")
    st.scatter_chart(data, x="Advertising_Spend", y="Sales", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Preview dataset"):
        st.dataframe(data.head(20), use_container_width=True)

# ---------- About ----------
else:
    st.subheader("About this project")
    st.markdown("""
    This application demonstrates a supervised machine learning regression workflow.
    The XGBoost model estimates the numerical **Sales** target from six business inputs.
    The trained model is stored locally as a reusable `.pkl` file and loaded by Streamlit.
    """)
    st.markdown("### Technologies")
    st.write("Python · Pandas · XGBoost · Scikit-learn · Joblib · Streamlit")
    st.markdown("### Project files")
    st.code("app.py\nsales_prediction_dataset.csv\nsales_xgboost_model.pkl\nrequirements.txt\nREADME.txt", language="text")
    st.markdown("### Important note")
    st.info("The supplied dataset is synthetic. Use real and appropriately validated business data before relying on forecasts for operational decisions.")

st.markdown("---")
st.caption("Sales Prediction using XGBoost • Academic project")
