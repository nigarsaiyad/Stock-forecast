```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Reliance Industries | AI Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1220 0%,
            #111827 50%,
            #0f172a 100%
        );
        color: white;
    }

    /* Main title */
    .hero-title {
        font-size: 46px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    .card-title {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .card-value {
        font-size: 28px;
        font-weight: 700;
    }

    .positive {
        color: #22c55e;
    }

    .negative {
        color: #ef4444;
    }

    /* Section headers */
    .section-header {
        font-size: 25px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        padding: 7px 15px;
        border-radius: 20px;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.3);
        color: #60a5fa;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1120;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">📈 Reliance Industries</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-Powered Stock Price Forecasting Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div style="text-align:center;">'
    '<span class="model-badge">ARIMA (1,0,2)</span>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load("arima_model.pkl")


try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Unable to load the trained ARIMA model."
    )

    st.exception(e)

    st.stop()


# =========================================================
# HISTORICAL DATA
# =========================================================

historical_data = pd.Series(
    model.data.orig_endog
)

historical_data = pd.to_numeric(
    historical_data,
    errors="coerce"
).dropna()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚙️ Forecast Controls")

forecast_period = st.sidebar.selectbox(
    "Forecast Horizon",
    [7, 14, 30],
    index=2
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Model")

st.sidebar.info(
    """
    **ARIMA(1,0,2)**

    AutoRegressive Integrated
    Moving Average model.

    Forecast type:
    **Multi-step forecast**
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Model Metrics")

st.sidebar.write("MAE: **6.213**")
st.sidebar.write("RMSE: **8.610**")
st.sidebar.write("MAPE: **1.608%**")
st.sidebar.write("R²: **0.967**")


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
# FORECAST DATAFRAME
# =========================================================

forecast_df = pd.DataFrame({

    "Forecast": forecast_mean.values,

    "Lower Bound":
        confidence_interval.iloc[:, 0].values,

    "Upper Bound":
        confidence_interval.iloc[:, 1].values

})


# =========================================================
# FORECAST INDEX
# =========================================================

if isinstance(historical_data.index, pd.DatetimeIndex):

    last_date = historical_data.index[-1]

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_period
    )

    forecast_df.index = future_dates

    forecast_df.index.name = "Date"

else:

    forecast_df.index = range(
        1,
        forecast_period + 1
    )

    forecast_df.index.name = "Forecast Period"


# =========================================================
# KEY VALUES
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
# DASHBOARD METRICS
# =========================================================

st.markdown(
    '<div class="section-header">📊 Market Snapshot</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">LATEST CLOSE</div>
            <div class="card-value">
                ₹{latest_price:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">
                {forecast_period}-DAY FORECAST
            </div>
            <div class="card-value">
                ₹{final_forecast:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    direction_class = (
        "positive"
        if expected_change >= 0
        else "negative"
    )

    direction_icon = (
        "📈"
        if expected_change >= 0
        else "📉"
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">EXPECTED CHANGE</div>
            <div class="card-value {direction_class}">
                {direction_icon} {expected_change:+.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">MODEL</div>
            <div class="card-value">
                ARIMA
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FORECAST SIGNAL
# =========================================================

st.markdown("---")

if expected_change > 0:

    st.success(
        f"📈 **Forecast Signal:** The model projects a "
        f"potential increase of **{expected_change:.2f}%** "
        f"over the selected horizon."
    )

elif expected_change < 0:

    st.warning(
        f"📉 **Forecast Signal:** The model projects a "
        f"potential decrease of **{abs(expected_change):.2f}%** "
        f"over the selected horizon."
    )

else:

    st.info(
        "➡️ **Forecast Signal:** The model projects "
        "relatively stable prices."
    )


# =========================================================
# CHART
# =========================================================

st.markdown(
    '<div class="section-header">'
    '📈 Historical Price & ARIMA Forecast'
    '</div>',
    unsafe_allow_html=True
)

fig, ax = plt.subplots(
    figsize=(15, 7)
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
    recent_history.index[-1],
    linestyle="--",
    linewidth=1.5,
    label="Forecast Start"
)

ax.set_title(
    f"Reliance Industries — {forecast_period} Day Forecast",
    fontsize=16
)

ax.set_xlabel("Date")

ax.set_ylabel("Price (₹)")

ax.legend()

ax.grid(
    alpha=0.25
)

plt.tight_layout()

st.pyplot(fig)


# =========================================================
# FORECAST TABLE
# =========================================================

st.markdown(
    '<div class="section-header">'
    '🔮 Detailed Forecast'
    '</div>',
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
# DOWNLOAD FORECAST
# =========================================================

csv = display_df.to_csv().encode("utf-8")

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv,
    file_name="reliance_arima_forecast.csv",
    mime="text/csv"
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-header">'
    '🎯 Model Performance'
    '</div>',
    unsafe_allow_html=True
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "MAE",
        "6.213"
    )

with p2:

    st.metric(
        "RMSE",
        "8.610"
    )

with p3:

    st.metric(
        "MAPE",
        "1.608%"
    )

with p4:

    st.metric(
        "R²",
        "0.967"
    )


# =========================================================
# MODEL EXPLANATION
# =========================================================

st.markdown("---")

with st.expander(
    "🧠 How does ARIMA work?"
):

    st.write(
        """
        **ARIMA** stands for **AutoRegressive Integrated
        Moving Average**.

        The selected model is **ARIMA(1,0,2)**:

        **p = 1 — AutoRegressive**

        The model uses one previous observation to help
        predict future values.

        **d = 0 — Integrated**

        No differencing was required for the final model.

        **q = 2 — Moving Average**

        The model incorporates information from two
        previous forecast errors.

        The model was evaluated using rolling
        one-step-ahead validation before being used
        for future forecasting.
        """
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

with st.expander(
    "📚 About this project"
):

    st.write(
        """
        This project demonstrates an end-to-end
        time-series forecasting workflow:

        • Historical stock-price analysis

        • Time-series preprocessing

        • ARIMA model selection

        • Rolling validation

        • Model evaluation

        • Future forecasting

        • Streamlit deployment

        The application is designed as an educational
        demonstration of machine-learning-based
        financial forecasting.
        """
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("---")

st.warning(
    "⚠️ **Disclaimer:** This application is for educational "
    "and analytical purposes only. Stock-price forecasts "
    "are inherently uncertain and should not be considered "
    "financial advice or a recommendation to buy or sell "
    "securities."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Built with Python • Statsmodels • Pandas •
        Matplotlib • Streamlit
        <br><br>
        Reliance Industries Stock Forecasting Project
    </div>
    """,
    unsafe_allow_html=True
)
```
