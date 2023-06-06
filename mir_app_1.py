import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st 

start_time = pd.to_datetime('2015-01-01')
end_time = pd.to_datetime('2022-03-01')

st.title('MAKE IT REAL')
st.title('STOCK PRICE PREDICTION')
user_input = st.text_input("Enter Stock Ticker")

data_acn = yf.download(user_input, start = start_time, end = end_time)

#Describing the data
st.subheader('Data from 2015 - 2022')
st.write(data_acn.describe())

st.subheader('Stock Price Sheet of the company from 2015 - 2022')
st.write(data_acn.head())

# Visualizations - entire dataset
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
acn_close = data_acn.reset_index()['Close']
acn_Dates = data_acn.reset_index()['Date']
plt.plot(acn_Dates, acn_close)
st.pyplot(fig)

st.subheader('Opening Price vs Time chart')
fig = plt.figure(figsize = (12,6))
acn_open = data_acn.reset_index()['Open']
acn_Dates = data_acn.reset_index()['Date']
plt.plot(acn_Dates, acn_open)
st.pyplot(fig)

st.subheader('Closing Price with EMA vs Time chart')
fig = plt.figure(figsize = (12,6))
acn_close = data_acn.reset_index()['Close']
acn_close_ema = acn_close.ewm(span=10).mean()
acn_Dates = data_acn.reset_index()['Date']
plt.plot(acn_Dates, acn_close)
plt.plot(acn_Dates, acn_close_ema)
plt.legend(['Actual Close Price','EMA Close Price'])
st.pyplot(fig)

st.subheader('Opening Price with EMA vs Time chart')
fig = plt.figure(figsize = (12,6))
acn_open = data_acn.reset_index()['Open']
acn_open_ema = acn_open.ewm(span=10).mean()
acn_Dates = data_acn.reset_index()['Date']
plt.plot(acn_Dates, acn_open)
plt.plot(acn_Dates, acn_open_ema)
plt.legend(['Actual Open Price','EMA Open Price'])
st.pyplot(fig)

data_acn.reset_index(inplace=True)
acn_open = data_acn.reset_index()['Open']
data_acn01 = data_acn.drop(['Open','High','Low','Adj Close','Volume'], axis=1)
# Taking last 60 days data--test set
last = data_acn01[len(data_acn01)-60:]
#dates fot test set
pred_dates01 = last.reset_index()['Date']
pred_dates = pred_dates01.tolist()
# actual values of test set
actual_yhat001 = last.reset_index()['Close']
actual_yhat = actual_yhat001.tolist()
data_acn01 = data_acn01.rename(columns={'Date':'ds', 'Close':'y'})
data_acn01 = data_acn01[:-60]

st.subheader('Data for training')
print('data_acn01')

# import the fbprophet library
from prophet import Prophet

# create the prophet object (model)
fbp = Prophet(daily_seasonality = True)
fbp.fit(data_acn01)

future = fbp.make_future_dataframe(periods=365)
forecast = fbp.predict(future)
print("Forecast data: ")
print(forecast)

# importing the plotting libraries
from prophet.plot import plot_plotly
#plot the data- PREDICTIONS
st.subheader('FORECAST MAP - Closing Price Predictions')
fig = plt.figure(figsize = (14,8))
# plot_plotly(fbp, forecast)
plt.plot(forecast['ds'].tolist(),forecast['yhat'].tolist())
plt.xlabel('Timeline from 2015 to 2022')
plt.ylabel('Closing Stock price')
# st.plotly_chart(fig)
st.pyplot(fig)

yhat_pred = forecast[1753:1813]['yhat'].tolist()

# rounding off the values upto 2 decimal places
L_actual=[]
for i in range(len(actual_yhat)):
  L_actual.append(round(actual_yhat[i], 3))
L_pred=[]
for i in range(len(yhat_pred)):
  L_pred.append(round(yhat_pred[i], 3))

## applying the Profit-signal algorithm
pnl_signal=[]
print("The working-day-wise Analysis from January 2022")
for i in range(len(L_pred)-1):
  x1 = L_pred[i+1] - L_actual[i]
  x2 = L_actual[i+1] - L_actual[i]
  x3 = L_actual[i]
  P = ((x1 * x2)/x3)
  pnl = np.sign(P)
  if pnl == 1.0 or pnl==1.00:
    print(f"The signal is to : BUY , for the DAY- {i+1}")
  else :
    print(f"The signal is to : SELL , for the DAY- {i+1}")
  pnl_signal.append(pnl)

actual_open = acn_open[len(acn_open)-60:].tolist()
L_open=[]
for i in range(len(actual_open)):
  L_open.append(round(actual_open[i], 3))

# profit calculation
profit_calc=[]
print("The working-day-wise Analysis from January 2022")
for i in range(len(pnl_signal)):
  y1 = L_actual[i] - L_open[i]
  y2 = pnl_signal[i]
  profit = y2 * y1
  if pnl_signal[i] == 1.0 :
    print(f"The signal is : BUY :{pnl}, for the DAY- {i+1} since profit :{-profit}")
  elif pnl_signal[i] == -1.0:
    print(f"The signal is : SELL :{pnl}, for the DAY- {i+1} since loss :{-profit}")
  profit_calc.append(profit)

zero_L=[]
for i in range(60):
  zero_L.append(0)

day_L=[]
for i in range(59):
  day_L.append(f"Day-{i+1}")

# Plotting the Profit-per day curve from Dates ---
st.subheader('Cumulative profit Graph')
fig = plt.figure(figsize = (14, 6))
plt.bar(day_L, profit_calc, color = 'maroon', width = 0.2)
plt.xlabel('Day 1-60')
plt.ylabel('Profit margin')
st.pyplot(fig)

# Cumulative Profit Calculation
st.subheader('Cumulative Profit for the Timeline of next 60 days')
cumu_pft= 0
for i in range(59):
  cumu_pft = cumu_pft+profit_calc[i]
st.write(cumu_pft)

print("The Cumulative Profit for the Timeline is : ", cumu_pft)
# plotting the profit curve
st.subheader('Profit Curve')
fig = plt.figure(figsize = (14,6))
plt.xlabel('60-Day-Timeline')
plt.ylabel('Profit')
plt.plot(profit_calc) 
plt.plot(zero_L)
plt.legend(["Profit Curve","Neutral Line"])
st.pyplot(fig)



