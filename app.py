import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Reliance Industries Stock Forecast",
    page_icon="💲",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    margin-bottom: 30px;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📈 Reliance Industries Stock Forecast</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'ARIMA(1,0,2) Time-Series Forecasting Model'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "arima_model.pkl"
    )

    return model


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained ARIMA model."
    )

    st.exception(e)

    st.stop()


# =========================================================
# GET HISTORICAL DATA FROM MODEL
# =========================================================

historical_data = model.data.orig_endog

historical_data = pd.Series(
    historical_data
)

historical_data = pd.to_numeric(
    historical_data,
    errors="coerce"
)

historical_data = historical_data.dropna()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Forecast Settings")

forecast_period = st.sidebar.selectbox(
    "Forecast Period",
    [7, 14, 30],
    index=2
)

st.sidebar.markdown("---")

st.sidebar.write(
    "### Model"
)

st.sidebar.write(
    "ARIMA(1,0,2)"
)

st.sidebar.write(
    "Forecast Type: One-step rolling validated model"
)

# =========================================================
# FUTURE DATES
# =========================================================

last_date = historical_data.index[-1]

future_dates = pd.bdate_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_period
)


# =========================================================
# FORECAST
# =========================================================

forecast_result = model.get_prediction(
    start=model.nobs,
    end=model.nobs + forecast_period - 1,
    index=future_dates
)

forecast_mean = forecast_result.predicted_mean

confidence_interval = forecast_result.conf_int(
    alpha=0.05
)

# =========================================================
# FORECAST DATAFRAME
# =========================================================

forecast_df = pd.DataFrame({

    "Forecast": forecast_mean.values,

    "Lower Bound":
        confidence_interval.iloc[:, 0].values,

    "Upper Bound":
        confidence_interval.iloc[:, 1].values

})


forecast_df.index = future_dates

forecast_df.index.name = "Date"


# =========================================================
# KEY METRICS
# =========================================================

latest_price = float(
    historical_data.iloc[-1]
)

final_forecast = float(
    forecast_df["Forecast"].iloc[-1]
)

expected_change = (
    (final_forecast - latest_price)
    / latest_price
) * 100


# =========================================================
# METRIC CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Latest Close",
        f"₹{latest_price:,.2f}"
    )


with col2:

    st.metric(
        f"{forecast_period}-Period Forecast",
        f"₹{final_forecast:,.2f}"
    )


with col3:

    st.metric(
        "Expected Change",
        f"{expected_change:.2f}%"
    )


with col4:

    st.metric(
        "Model",
        "ARIMA(1,0,2)"
    )


# =========================================================
# FORECAST INTERPRETATION
# =========================================================

st.markdown("---")

if expected_change > 0:

    st.success(
        f"📈 The model forecasts an increase of "
        f"{expected_change:.2f}% over the selected "
        f"forecast horizon."
    )

elif expected_change < 0:

    st.warning(
        f"📉 The model forecasts a decrease of "
        f"{abs(expected_change):.2f}% over the selected "
        f"forecast horizon."
    )

else:

    st.info(
        "The model forecasts relatively stable prices."
    )


# =========================================================
# FORECAST CHART
# =========================================================

st.subheader(
    "📊 Historical Price & Future Forecast"
)


fig, ax = plt.subplots(
    figsize=(14, 6)
)


# Recent historical prices
recent_history = historical_data.tail(120)


ax.plot(
    recent_history.index,
    recent_history.values,
    label="Historical Close"
)


# Forecast
ax.plot(
    forecast_df.index,
    forecast_df["Forecast"],
    label="ARIMA Forecast",
    linewidth=2
)


# Confidence interval
ax.fill_between(
    forecast_df.index,
    forecast_df["Lower Bound"],
    forecast_df["Upper Bound"],
    alpha=0.20,
    label="95% Confidence Interval"
)


# Forecast starting line
ax.axvline(
    historical_data.index[-1],
    linestyle="--",
    alpha=0.7,
    label="Forecast Start"
)


ax.set_title(
    f"ARIMA(1,0,2) - {forecast_period} Period Forecast"
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Closing Price"
)

ax.legend()

ax.grid(
    alpha=0.3
)

plt.tight_layout()


st.pyplot(fig)


# =========================================================
# FORECAST TABLE
# =========================================================

st.subheader(
    "🔮 Forecast Details"
)


display_df = forecast_df.copy()

display_df["Forecast"] = (
    display_df["Forecast"]
    .round(2)
)

display_df["Lower Bound"] = (
    display_df["Lower Bound"]
    .round(2)
)

display_df["Upper Bound"] = (
    display_df["Upper Bound"]
    .round(2)
)


st.dataframe(
    display_df,
    use_container_width=True
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.subheader(
    "🤖 Model Performance"
)


performance_col1, performance_col2, performance_col3, performance_col4 = st.columns(4)


with performance_col1:

    st.metric(
        "MAE",
        "6.192"
    )


with performance_col2:

    st.metric(
        "RMSE",
        "8.595"
    )


with performance_col3:

    st.metric(
        "MAPE",
        "1.601%"
    )


with performance_col4:

    st.metric(
        "R²",
        "0.967"
    )


# =========================================================
# MODEL EXPLANATION
# =========================================================

st.markdown("---")

st.subheader(
    "🤖 About the Model"
)

st.write(
    """
    **ARIMA (AutoRegressive Integrated Moving Average)** is a
    time-series forecasting model that learns patterns from
    historical stock prices.

    The selected model is **ARIMA(1,0,2)**:

    - **1** → uses one autoregressive component
    - **0** → no differencing was required in the final model
    - **2** → uses two moving-average components

    The model was selected after comparing multiple ARIMA
    configurations and validated using rolling one-step-ahead
    forecasting.
    """
)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.caption(
    "⚠️ This application is for educational and analytical "
    "purposes only. Stock-market forecasts are uncertain and "
    "should not be considered financial advice."
)
