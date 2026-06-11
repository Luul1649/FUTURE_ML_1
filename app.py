import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.express as px

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.title("📈 Sales & Demand Forecasting Dashboard")

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(r"Sample - Superstore.csv", encoding="latin1")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Convert Date
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Aggregate Daily Sales
    sales = df.groupby('Order Date')['Sales'].sum().reset_index()

    sales.columns = ['ds', 'y']

    st.subheader("Daily Sales Data")
    st.dataframe(sales.head())

    # Sales Trend
    st.subheader("Sales Trend")

    fig = px.line(
        sales,
        x='ds',
        y='y',
        title='Daily Sales Trend'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Horizon
    forecast_days = st.slider(
        "Select Forecast Period (Days)",
        min_value=30,
        max_value=365,
        value=90
    )

    if st.button("Generate Forecast"):

        with st.spinner("Training Prophet Model..."):

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True
            )

            model.fit(sales)

            future = model.make_future_dataframe(
                periods=forecast_days
            )

            forecast = model.predict(future)

        st.success("Forecast Generated Successfully!")

        # Forecast Table
        st.subheader("Forecast Results")

        st.dataframe(
            forecast[
                ['ds',
                 'yhat',
                 'yhat_lower',
                 'yhat_upper']
            ].tail(forecast_days)
        )

        # Forecast Plot
        st.subheader("Forecast Visualization")

        fig1 = model.plot(forecast)

        st.pyplot(fig1)

        # Components
        st.subheader("Trend & Seasonality")

        fig2 = model.plot_components(forecast)

        st.pyplot(fig2)

        # Download
        csv = forecast.to_csv(index=False)

        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv"
        )

else:

    st.info("Upload the Superstore CSV dataset to begin.")
