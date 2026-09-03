# 📈 Stock Price Forecasting & Time-Series Analysis

An end-to-end **stock price forecasting project** that combines statistical time-series models, machine learning, and deep learning to predict future stock prices.

The project covers the complete data science workflow:

**EDA → Feature Engineering → Time-Series Modeling → Model Evaluation → Forecasting → Deployment**

---

## 🚀 Project Overview

This project analyzes historical stock-price data and develops multiple forecasting models to determine which approach provides the most accurate predictions.

The following models were evaluated:

* Naive Forecast
* Linear Regression
* Lasso Regression
* Random Forest
* XGBoost
* LSTM
* ARIMA
* SARIMA
* SARIMAX
* Holt-Winters
* Prophet

After extensive experimentation, **Lasso Regression** achieved the strongest overall performance among the tested models, while **ARIMA(1,0,2)** was selected as the final statistical time-series forecasting model for deployment.

---

## 🎯 Objectives

The main objectives of this project were to:

* Analyze historical stock-price trends
* Understand time-series behavior
* Perform exploratory data analysis
* Engineer lag and moving-average features
* Build multiple forecasting models
* Compare models using consistent evaluation metrics
* Identify the best-performing models
* Generate future stock-price forecasts
* Deploy the forecasting model using Streamlit

---

# 📊 Exploratory Data Analysis

The dataset contains historical stock-market information including:

* Open
* High
* Low
* Close
* Adjusted Close
* Volume

Additional features were created during preprocessing and feature engineering.

### EDA included:

* Historical closing-price trends
* Trading volume analysis
* Daily return analysis
* Moving averages
* Volatility analysis
* Correlation analysis
* Distribution analysis
* Outlier investigation
* Actual vs predicted visualizations

---

# ⚙️ Feature Engineering

Several time-series features were created to capture historical price behavior.

### Lag Features

Lag variables represent previous observations.

For example:

```text
Close_Lag_1  → Previous day's closing price
Close_Lag_2  → Closing price two days ago
Close_Lag_3  → Closing price three days ago
Close_Lag_5  → Closing price five days ago
Close_Lag_10 → Closing price ten days ago
```

### Moving Averages

The project also uses:

```text
MA_5
MA_10
MA_20
```

These smooth short-term price fluctuations and help identify trends.

### Volatility

A 10-period volatility feature was created to measure how much the stock price fluctuates.

---

# 🤖 Models

## 1. Naive Forecast

The Naive model uses the previous observed closing price as the prediction for the next period.

It provides an important baseline against which more advanced models can be compared.

---

## 2. Linear Regression

Linear Regression predicts the next closing price using engineered historical features such as:

* Lagged prices
* Moving averages
* Volatility
* Trading volume

---

## 3. Lasso Regression

Lasso Regression adds regularization to Linear Regression and can automatically reduce the influence of less useful features.

Final evaluation:

| Metric |      Score |
| ------ | ---------: |
| MAE    |  **6.388** |
| RMSE   |  **8.797** |
| MAPE   | **1.649%** |
| R²     |  **0.963** |

Lasso produced the strongest overall predictive performance in the machine-learning comparison.

---

## 4. Random Forest

Random Forest uses an ensemble of decision trees to capture nonlinear relationships between historical features and future prices.

---

## 5. XGBoost

XGBoost is a gradient-boosting algorithm designed to model complex nonlinear relationships.

---

## 6. LSTM

Long Short-Term Memory neural networks were tested because they are specifically designed to learn patterns from sequential data.

The LSTM model provided a useful deep-learning comparison but did not outperform the simpler statistical and machine-learning approaches in this dataset.

---

# 📈 Statistical Time-Series Models

## ARIMA

ARIMA stands for:

**AutoRegressive Integrated Moving Average**

It models relationships between observations over time.

The final selected ARIMA model was:

```text
ARIMA(1,0,2)
```

### Rolling Forecast Performance

| Metric |      Score |
| ------ | ---------: |
| MAE    |  **6.192** |
| RMSE   |  **8.595** |
| MAPE   | **1.601%** |
| R²     |  **0.967** |

The model was evaluated using **rolling one-step-ahead forecasting**, where each actual observation is added back into the historical series before forecasting the next observation.

This provides a more realistic evaluation of how the model would behave in practice.

---

# 📊 SARIMA / SARIMAX

SARIMA and SARIMAX extend ARIMA by allowing the model to capture seasonal patterns and, in the case of SARIMAX, additional explanatory variables.

Several combinations of:

* AR terms
* Differencing
* MA terms
* Seasonal components

were evaluated.

The best SARIMAX configuration achieved:

| Metric |     Score |
| ------ | --------: |
| MAE    | **20.27** |
| RMSE   | **25.10** |
| MAPE   | **5.22%** |
| R²     | **0.696** |

Although SARIMAX was useful for experimentation, it did not outperform the final ARIMA or Lasso models.

---

# 📉 Holt-Winters

Holt-Winters exponential smoothing was evaluated to capture:

* Level
* Trend
* Seasonality

The model was included as another classical forecasting approach for comparison.

---

# 🔮 Prophet

Prophet was also evaluated as an alternative forecasting framework designed to handle trend and seasonal patterns in time-series data.

---

# 🏆 Model Comparison

The major model comparison was:

| Model               |       MAE |      RMSE |       MAPE |        R² |
| ------------------- | --------: | --------: | ---------: | --------: |
| 🥇 Lasso Regression | **6.388** | **8.797** |     1.649% | **0.963** |
| Naive Forecast      |     6.657 |     9.161 | **1.632%** |     0.925 |
| Linear Regression   |     8.066 |    10.460 |     1.978% |     0.903 |
| XGBoost             |    10.329 |    13.110 |     2.631% |     0.917 |
| Random Forest       |    11.957 |    14.810 |     3.037% |     0.894 |
| LSTM                |    16.076 |    18.911 |     3.910% |     0.693 |
| SARIMAX             |    20.274 |    25.096 |     5.220% |     0.696 |

### Key Finding

**Lasso Regression** achieved the best overall machine-learning performance.

However, the final deployment focuses on **ARIMA(1,0,2)** because it provides a dedicated statistical time-series forecasting approach and achieved excellent rolling-forecast performance.

---

# 🥇 Final ARIMA Model

The final ARIMA model is:

```text
ARIMA(1,0,2)
```

Performance:

```text
MAE  : 6.192
RMSE : 8.595
MAPE : 1.601%
R²   : 0.967
```

The model was refitted using the complete historical closing-price series before generating future forecasts.

---

# 🔮 Future Forecast

The deployment generates future forecasts along with confidence intervals.

Example output:

| Date         | Forecast | Lower Bound | Upper Bound |
| ------------ | -------: | ----------: | ----------: |
| Future Day 1 |   361.36 |      338.31 |      384.41 |
| Future Day 2 |   361.47 |      328.43 |      394.51 |
| Future Day 3 |   361.71 |      321.47 |      401.94 |
| ...          |      ... |         ... |         ... |

The confidence interval communicates the uncertainty associated with the forecast.

---

# 🌐 Deployment

The forecasting application is built using **Streamlit**.

The deployed application provides:

* 📈 Historical stock-price visualization
* 🔮 Future price forecasting
* 📊 Forecast table
* 📉 Confidence intervals
* 💰 Latest closing price
* 🎯 Forecasted price
* 📈 Expected percentage change
* 🤖 Model information
* 📊 Model performance metrics

### Application Workflow

```text
Historical Stock Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
ARIMA(1,0,2)
        ↓
Model Forecast
        ↓
Confidence Intervals
        ↓
Streamlit Dashboard
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib

### Machine Learning

* Scikit-learn
* Random Forest
* XGBoost

### Deep Learning

* TensorFlow / Keras
* LSTM

### Time-Series

* Statsmodels
* ARIMA
* SARIMA
* SARIMAX
* Holt-Winters
* Prophet

### Deployment

* Streamlit

### Model Serialization

* Joblib

---

# 📁 Project Structure

```text
stock-forecast/
│
├── app.py
├── final_arima_102.pkl
├── arima_future_forecast.csv
├── requirements.txt
├── README.md
│
└── notebooks/
    └── stock_forecasting.ipynb
```

---

# ▶️ Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd stock-forecast
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 📌 Key Takeaways

### 1. Simple models can be powerful

The Naive Forecast performed surprisingly well, showing that stock prices contain strong short-term persistence.

### 2. Lasso performed best among the machine-learning models

Lasso achieved an R² of approximately **0.963** while maintaining low MAE and RMSE.

### 3. ARIMA performed strongly when evaluated correctly

The rolling ARIMA(1,0,2) forecast achieved approximately:

**96.7% R²**

with only:

**1.60% MAPE**

### 4. Complex models did not automatically perform better

Random Forest, XGBoost, LSTM, and SARIMAX did not outperform the simpler Lasso and ARIMA approaches on this dataset.

This highlights an important machine-learning principle:

> **Model complexity does not guarantee better predictive performance.**

---

# ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes only**.

Stock prices are influenced by many unpredictable factors, including market sentiment, economic conditions, company news, geopolitical events, and broader market movements.

The forecasts generated by this application should **not be considered financial advice or a recommendation to buy or sell securities**.

---

# 👩‍💻 Author

Nigar Saiyad
