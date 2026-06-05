import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Superstore Demand Forecaster",
    page_icon="📈",
    layout="wide"
)

# 2. Application Header
st.title("📈 AI Sales Forecasting & Inventory Control Center")
st.markdown("This dashboard leverages a trained **Gradient Boosting Engine** to project demand and optimize warehouse safety stock.")

# 3. Load the Serialized Pipeline Bundle
@st.cache_resource
def load_models():
    try:
        with open("superstore_forecaster_payload.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("🚨 Error: 'superstore_forecaster_payload.pkl' not found! Please run your training script first to generate the model file.")
        return None

pipeline_bundle = load_models()

if pipeline_bundle:
    # 4. Sidebar Controls
    st.sidebar.header("🛠️ Operational Parameters")
    
    # Category Selector
    category_list = list(pipeline_bundle.keys())
    selected_cat = st.sidebar.selectbox("Select Business Unit / Category", category_list)
    
    # Mock Inventory Inputs
    st.sidebar.subheader("Warehouse Inventory Check")
    current_stock = st.sidebar.number_input(
        "Current On-Hand Stock Capital ($)", 
        min_value=0.0, 
        value=3500.0, 
        step=100.0
    )
    
    # Upcoming Week Parameters
    st.sidebar.subheader("Upcoming Week Context")
    is_holiday = st.sidebar.checkbox("Is the upcoming week a Holiday Week?", value=False)
    
    # Extract selected model assets
    cat_assets = pipeline_bundle[selected_cat]
    model = cat_assets['model_object']
    error_margin = cat_assets['model_mape']
    
    # 5. Live Simulation Inputs
    st.header(f"📊 Live Demand Projections: {selected_cat}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        last_week_sales = st.number_input("Sales from Last Week ($)", min_value=0.0, value=4000.0, step=100.0)
    with col2:
        two_weeks_ago_sales = st.number_input("Sales from 2 Weeks Ago ($)", min_value=0.0, value=3800.0, step=100.0)
    with col3:
        rolling_avg = st.number_input("4-Week Running Average ($)", min_value=0.0, value=3900.0, step=100.0)
        
    # Standard Date context fallback for live mock simulation
    current_year = 2026
    current_month = 6
    current_week = 23

    # Construct the input feature DataFrame matching the exact trained columns
    live_features = pd.DataFrame([{
        'Year': current_year,
        'Month': current_month,
        'Week_of_Year': current_week,
        'Is_Holiday_Week': int(is_holiday),
        'Sales_Last_Week': last_week_sales,
        'Sales_2Weeks_Ago': two_weeks_ago_sales,
        'Rolling_Avg_Month': rolling_avg
    }])

    # 6. Generate Forecast and Safety Calculations
    ordered_cols = cat_assets['features_ordered']
    prediction = model.predict(live_features[ordered_cols])[0]
    safety_stock = prediction * error_margin
    total_required_capital = prediction + safety_stock

    # 7. Display Executive Summary Metrics
    st.markdown("---")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    m_col1.metric(
        label="🎯 Predicted Sales Demand", 
        value=f"${prediction:,.2f}",
        help="The core projection computed by the ML model."
    )
    m_col2.metric(
        label="🛡️ Required Safety Buffer", 
        value=f"${safety_stock:,.2f}",
        delta=f"{(error_margin*100):.1f}% Error Margin",
        delta_color="inverse",
        help="Extra inventory cushion needed to guard against model error variability."
    )
    m_col3.metric(
        label="📋 Total Supply Coverage Needed", 
        value=f"${total_required_capital:,.2f}",
        help="Sum of predicted demand and safety stock required to avoid stockouts."
    )

    # 8. Operational Exception Alerter Component
    st.subheader("🚨 Warehouse Logistics Risk Analysis")
    
    if current_stock < total_required_capital:
        shortfall = total_required_capital - current_stock
        st.error(
            f"**CRITICAL DEFICIT DETECTED:** Current on-hand stock (**${current_stock:,.2f}**) is insufficient "
            f"to cover upcoming demand and safety parameters. You are exposed to stockout liabilities.\n\n"
            f"📥 **Supply Chain Action:** Issue a procurement order for an additional **${shortfall:,.2f}** immediately."
        )
    else:
        surplus = current_stock - total_required_capital
        st.success(
            f"**STOCK LEVEL SECURE:** Warehouse capital (**${current_stock:,.2f}**) safely exceeds the required "
            f"coverage boundary. You have a **${surplus:,.2f}** buffer above peak variance requirements."
        )
