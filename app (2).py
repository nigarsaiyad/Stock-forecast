import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Reliance Industries | AI Stock Forecast",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main title */

.hero-title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    color: #9aa4b2;
    margin-bottom: 35px;
}

/* Metric cards */

.metric-card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06);
    text-align: center;
}

.metric-label {
    font-size: 14px;
    color: #9aa4b2;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 28px;
    font-weight: 750;
}

.metric-positive {
    color: #21c77a;
}

.metric-negative {
    color: #ff5c5c;
}

/* Section headings */

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 20px;
}

/* Model badge */

.model-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    border: 1px solid rgba(128,128,128,0.35);
    font-weight: 600;
}

/* Footer */

.footer {
    text-align: center;
    color: #8b949e;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load("arima_model.pkl")


try:

    model = load_model()

except Exception as e:

    st.error("❌ Unable to load the trained ARIMA model.")

    st.exception(e)

    st.stop()


# =========================================================
# HISTORICAL DATA
# =========================================================

original_data = model.data.orig_endog

historical_values = (
    pd.Series(original_data)
    .astype(float)
    .values
)

# Reconstruct trading-day index.
#
# The model contains the historical observations but
# statsmodels did not preserve a supported DatetimeIndex.

historical_dates = pd.bdate_range(
    end="2023-10-16",
    periods=len(historical_values)
)

historical_data = pd.Series(
    historical_values,
    index=historical_dates,
    name="Close"
)

historical_data = pd.to_numeric(
    historical_data,
    errors="coerce"
).dropna()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🚀 ARIMA Forecast Control")

st.sidebar.markdown("### Forecast Horizon")

forecast_period = st.sidebar.select_slider(
    "Select forecast period",
    options=[7, 14, 30],
    value=30
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Model")

st.sidebar.info(
    """
    **ARIMA(1,0,2)**

    AutoRegressive Integrated Moving Average

    Validation:
    Rolling one-step-ahead forecasting
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Project")

st.sidebar.write(
    "Reliance Industries Stock Forecast"
)

st.sidebar.write(
    "Historical data: 2020–2023"
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero-title">📈 Reliance Industries Stock Forecast</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-powered time-series forecasting using ARIMA(1,0,2)'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div style="text-align:center;">'
    '<span class="model-badge">ARIMA(1,0,2)</span>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FORECAST
# =========================================================

forecast_result = model.get_forecast(
    steps=forecast_period
)

forecast_mean = forecast_result.predicted_mean

confidence_interval = forecast_result.conf_int(
    alpha=0.05
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

forecast_low = float(
    forecast_df["Lower Bound"].iloc[-1]
)

forecast_high = float(
    forecast_df["Upper Bound"].iloc[-1]
)


# =========================================================
# KPI CARDS
# =========================================================

st.markdown("### 📊 Forecast Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Latest Close</div>
            <div class="metric-value">
                ₹{latest_price:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {forecast_period}-Period Forecast
            </div>
            <div class="metric-value">
                ₹{final_forecast:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    change_class = (
        "metric-positive"
        if expected_change >= 0
        else "metric-negative"
    )

    arrow = "▲" if expected_change >= 0 else "▼"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Expected Change
            </div>
            <div class="metric-value {change_class}">
                {arrow} {abs(expected_change):.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Forecast Range
            </div>
            <div class="metric-value">
                ₹{forecast_low:,.0f} – ₹{forecast_high:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FORECAST SUMMARY
# =========================================================

st.markdown("---")

if expected_change > 0:

    st.success(
        f"📈 **Forecast outlook:** ARIMA predicts an increase "
        f"of **{expected_change:.2f}%** over the selected "
        f"{forecast_period}-period horizon."
    )

elif expected_change < 0:

    st.warning(
        f"📉 **Forecast outlook:** ARIMA predicts a decrease "
        f"of **{abs(expected_change):.2f}%** over the selected "
        f"{forecast_period}-period horizon."
    )

else:

    st.info(
        "➡️ **Forecast outlook:** The model predicts "
        "relatively stable prices."
    )


# =========================================================
# CHART
# =========================================================

st.markdown(
    '<div class="section-title">📈 Historical Price & Forecast</div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(14, 6)
)

recent_history = historical_data.tail(120)

ax.plot(
    recent_history.index,
    recent_history.values,
    label="Historical Close",
    linewidth=2
)

ax.plot(
    forecast_df.index,
    forecast_df["Forecast"],
    label="ARIMA Forecast",
    linewidth=2.5
)

ax.fill_between(
    forecast_df.index,
    forecast_df["Lower Bound"],
    forecast_df["Upper Bound"],
    alpha=0.20,
    label="95% Confidence Interval"
)

ax.axvline(
    historical_data.index[-1],
    linestyle="--",
    linewidth=1.5,
    label="Forecast Start"
)

ax.set_title(
    f"ARIMA(1,0,2) — {forecast_period}-Period Forecast",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Date")

ax.set_ylabel("Closing Price (₹)")

ax.legend()

ax.grid(
    alpha=0.25
)

plt.tight_layout()

st.pyplot(fig)


# =========================================================
# FORECAST TABLE
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔮 Forecast Details</div>',
    unsafe_allow_html=True
)

display_df = forecast_df.copy()

display_df["Forecast"] = (
    display_df["Forecast"].round(2)
)

display_df["Lower Bound"] = (
    display_df["Lower Bound"].round(2)
)

display_df["Upper Bound"] = (
    display_df["Upper Bound"].round(2)
)

st.dataframe(
    display_df,
    use_container_width=True
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

perf1, perf2, perf3, perf4 = st.columns(4)


with perf1:

    st.metric(
        "MAE",
        "6.192"
    )


with perf2:

    st.metric(
        "RMSE",
        "8.595"
    )


with perf3:

    st.metric(
        "MAPE",
        "1.601%"
    )


with perf4:

    st.metric(
        "R²",
        "0.967"
    )


st.caption(
    "Performance metrics are based on rolling one-step-ahead "
    "validation of the ARIMA model."
)


# =========================================================
# MODEL EXPLANATION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 Understanding the Model</div>',
    unsafe_allow_html=True
)

with st.expander(
    "What does ARIMA(1,0,2) mean?"
):

    st.markdown(
        """
        **ARIMA** stands for:

        **AutoRegressive Integrated Moving Average**

        The selected configuration is:

        **ARIMA(1,0,2)**

        **p = 1 — AutoRegressive component**

        The model uses one previous observation to help
        predict the current value.

        **d = 0 — Differencing**

        No differencing was required for the final model,
        meaning the original price series was modeled directly.

        **q = 2 — Moving Average component**

        The model incorporates information from the previous
        two forecast errors.

        The model was selected after comparing multiple
        ARIMA configurations and evaluating them using
        rolling one-step-ahead forecasting.
        """
    )


with st.expander(
    "Why was ARIMA selected?"
):

    st.markdown(
        """
        ARIMA was selected because it is specifically designed
        for time-series forecasting and can model temporal
        dependencies in historical stock prices.

        The model was evaluated using rolling validation rather
        than relying only on training-set performance.

        Final validation performance:

        - **MAE:** 6.192
        - **RMSE:** 8.595
        - **MAPE:** 1.601%
        - **R²:** 0.967

        These metrics indicate that the model provides a strong
        fit on the validation period, although stock prices
        remain inherently difficult to forecast.
        """
    )


with st.expander(
    "How should the forecast be interpreted?"
):

    st.markdown(
        """
        The forecast represents the model's estimated future
        closing prices based on historical price behavior.

        The shaded area represents the **95% confidence interval**.

        A wider interval indicates greater uncertainty.

        The forecast should therefore be interpreted as a
        statistical estimate rather than a guaranteed future
        price.
        """
    )


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🚀 Project Summary</div>',
    unsafe_allow_html=True
)

summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.markdown(
        """
        ### 📌 Machine Learning Pipeline

        **Historical Data → EDA → Time-Series Analysis →  
        ARIMA Model Selection → Rolling Validation →  
        Forecast → Streamlit Deployment**
        """
    )


with summary_col2:

    st.markdown(
        """
        ### 🛠️ Technology Stack

        - Python
        - Pandas
        - NumPy
        - Statsmodels
        - Joblib
        - Matplotlib
        - Streamlit

        **Model:** ARIMA(1,0,2)
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.warning(
    "⚠️ This application is for educational and analytical "
    "purposes only. Stock-market forecasts are uncertain and "
    "should not be considered financial advice."
)

st.markdown(
    '<div class="footer">'
    'Built with Python • Statsmodels • Streamlit • ARIMA'
    '</div>',
    unsafe_allow_html=True
)
```
