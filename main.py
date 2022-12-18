import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Variables
today = date.today()
stock = "GOOG"
start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")  # A year ago
end_date = today.strftime("%Y-%m-%d")  # Today

# Prepare Data
data = yf.download(stock, start=start_date, end=end_date, progress=False)
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
data = data[["Date", "Close"]]

# First View
plt.style.use("fivethirtyeight")
plt.figure(figsize=(15, 10))
plt.plot(data["Date"], data["Close"])
plt.show()

# Stationarity Test
result = seasonal_decompose(data["Close"], model="multiplicative", period=30)
fig = result.plot()
fig.set_size_inches(15, 10)
plt.show()


def ADF_test(dataset):
    test = adfuller(dataset, autolag="AIC")
    print("1. ADF : ", test[0])
    print("2. P-Value : ", test[1])
    print("3. Num Of Lags : ", test[2])
    print("4. Num Of Observations Used For ADF Regression:", test[3])
    print("5. Critical Values :")

    for key, val in test[4].items():
        print("\t", key, ": ", val)

    if test[1] < 0.05:
        return print("Result: Data is stationary. Cannot be used on the analysis.")
    else:
        return print("Result: Data is not stationary. Can be used on the analysis.")


ADF_test(data.Close)

# Prepare to ARIMA

# p value
pd.plotting.autocorrelation_plot(data["Close"])
plt.show()
# q value
plot_pacf(data["Close"], lags=100)
plt.show()

p = 5
d = 1
q = 2

# ARIMA
model = ARIMA(data["Close"], order=(p, d, q))
fitted = model.fit()
print(fitted.summary())

# Predict
predictions = fitted.predict()
print(predictions)

# SARIMA
model = sm.tsa.statespace.SARIMAX(
    data["Close"], order=(p, d, q), seasonal_order=(p, d, q, 12)
)
model = model.fit()
print(model.summary())

# Predict
predictions = model.predict(len(data), len(data) + 10)
print(predictions)

# Visualize
data["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")
plt.show()
