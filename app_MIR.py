import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st 

start_time = pd.to_datetime('2015-01-01')
end_time = pd.to_datetime('2021-03-01')

st.title('Make It Real')
st.title('Stock Price Prediction')
user_input = st.text_input("Enter Stock Ticker")

data_acn = yf.download(user_input, start = start_time, end = end_time)

#Describing the data
st.subheader('Data from 2015 - 2020')
st.write(data_acn.describe())

st.subheader('Stock Price Sheet of the company from 2015 - 2020')
st.write(data_acn.head())

acn_close = data_acn.reset_index()['Close']
acn_close_ema = acn_close.ewm(span=10).mean()

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


# Scaling up the given data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
acn_close = scaler.fit_transform(np.array(acn_close).reshape(-1,1))

Date01_acn= acn_Dates.to_numpy()
acn_date = Date01_acn.reshape(1549, 1)

#splitting dataset into train, validation and test datasets of Accenture(Close)
training_size = int(len(acn_close)*0.65)
t_size = len(acn_close)-training_size
valid_size = int(t_size*0.7)
test_size = int(t_size-valid_size)
train_data_acn_close, valid_data_acn_close, test_data_acn_close = acn_close[0:training_size, :], acn_close[training_size:(training_size + valid_size), :], acn_close[(training_size + valid_size):len(acn_close), :]

#splitting dataset into train, validation and test datasets of Accenture(Open)
#training_size_o = int(len(acn_open)*0.65)
#t_size_o = len(acn_open)-training_size_o
#valid_size_o = int(t_size_o*0.7)
#test_size_o = int(t_size_o-valid_size_o)
#train_data_acn_open, valid_data_acn_open, test_data_acn_open = acn_open[0:training_size_o, :], acn_open[training_size_o:(training_size_o + valid_size_o), :], acn_open[(training_size_o + valid_size_o):len(acn_open), :]


# defining a function to add the required timesteps
def create_dataset(dataset, time_steps=1):
  dataX, dataY= [], []
  for i in range(len(dataset)-time_steps-1):
    a=dataset[i:(i+time_steps), 0]
    dataX.append(a)
    dataY.append(dataset[i+time_steps, 0])
  return np.array(dataX), np.array(dataY)
# taking timesteps and change it accordingly during fine-tuning
time_steps =100
# fittting the train and test sets (Close)
X_train_acn_close, y_train_acn_close = create_dataset(train_data_acn_close, time_steps)
X_valid_acn_close, y_valid_acn_close = create_dataset(valid_data_acn_close,time_steps)
X_test_acn_close, y_test_acn_close = create_dataset(test_data_acn_close, time_steps)
# reshaping inputs into 3 dimensions 
X_train_acn_close = X_train_acn_close.reshape(X_train_acn_close.shape[0], X_train_acn_close.shape[1],1)
X_valid_acn_close = X_valid_acn_close.reshape(X_valid_acn_close.shape[0], X_valid_acn_close.shape[1],1)
X_test_acn_close = X_test_acn_close.reshape(X_test_acn_close.shape[0], X_test_acn_close.shape[1],1)

# fittting the train and test sets for Accenture (Open)
#X_train_acn_open, y_train_acn_open = create_dataset(train_data_acn_open, time_steps)
#X_valid_acn_open, y_valid_acn_open = create_dataset(valid_data_acn_open, time_steps)
#X_test_acn_open, y_test_acn_open = create_dataset(test_data_acn_open, time_steps)
# reshaping inputs into 3 dimensions 
#X_train_acn_open = X_train_acn_open.reshape(X_train_acn_open.shape[0], X_train_acn_open.shape[1],1)
#X_valid_acn_open = X_valid_acn_open.reshape(X_valid_acn_open.shape[0], X_valid_acn_open.shape[1],1)
#X_test_acn_open = X_test_acn_open.reshape(X_test_acn_open.shape[0], X_test_acn_open.shape[1],1)

# Load my model
model2 = load_model('keras_model_1.h5')

train_predict_acn_close=model2.predict(X_train_acn_close)
valid_predict_acn_close=model2.predict(X_valid_acn_close)
test_predict_acn_close=model2.predict(X_test_acn_close)
# ReverseScaling to original form 
train_predict_acn_close=scaler.inverse_transform(train_predict_acn_close)
valid_predict_acn_close=scaler.inverse_transform(valid_predict_acn_close)
test_predict_acn_close=scaler.inverse_transform(test_predict_acn_close)

# Visualizations -Checking the valid set of data with the predicted values for the valid set
valid_predict_acn_close=model2.predict(X_valid_acn_close).flatten()
st.subheader('ValidationSet- Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(acn_date[1006:1285], valid_predict_acn_close)
plt.plot(acn_date[1006:1285], y_valid_acn_close)
plt.legend(['ValidationData Predictions', 'ValidationData Observations' ])
st.pyplot(fig)

# Checking the train set of data with the predicted values for the train set 
test_predict_acn_close=model2.predict(X_test_acn_close).flatten()
st.subheader('Predictions-Test- Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(acn_date[1386:1448], test_predict_acn_close)
plt.plot(acn_date[1386:1448], y_test_acn_close)
plt.legend(['Test Predictions','Test Observations'])
st.pyplot(fig)

# Final Predictions for profit and loss

y_test_acn_close= y_test_acn_close.reshape(62,1)
y_test_acn_close=scaler.inverse_transform(y_test_acn_close)
y_test_acn_close= y_test_acn_close.flatten()
test_predict_acn_close=test_predict_acn_close.flatten()
## converting into list
y_test_acn_close_L=y_test_acn_close.tolist()
test_predict_acn_close_L=test_predict_acn_close.tolist()

#y_test_acn_open = y_test_acn_open.reshape(62,1)
#y_test_acn_open = scaler.inverse_transform(y_test_acn_open )
# converting both to lists
#y_test_acn_close= y_test_acn_close.flatten()
#y_test_acn_close_L=y_test_acn_close.tolist()
#y_test_acn_open= y_test_acn_open.flatten()
#y_test_acn_open_L=y_test_acn_open.tolist()

# rounding off the values upto 2 decimal places
L_test_new=[]
for i in range(len(y_test_acn_close_L)):
  L_test_new.append(round(y_test_acn_close_L[i], 2))
L_pred_new=[]
for i in range(len(test_predict_acn_close_L)):
  L_pred_new.append(round(test_predict_acn_close_L[i], 2))


## Applying and Listing by the Profit-signal algorithm
pnl_signal=[]
st.subheader('Listing out the predictions for Profit or Loss')
for i in range(len(L_pred_new)-1):
  x1 = L_pred_new[i+1] - L_test_new[i]
  x2 = L_test_new[i+1] - L_test_new[i]
  x3 = L_test_new[i]
  P = ((x1 * x2)/x3)
  pnl = np.sign(P)
  if pnl == 1.0 or pnl==1.00:
    st.write(f"The signal for the day - {acn_date[i+1386]} is to : BUY :{pnl}")
  else :
    st.write(f"The signal for the day - {acn_date[i+1386]} is to : SELL :{pnl}")
  pnl_signal.append(pnl)



#rounding off y_test_acn_close and y_test_acn_open upto 2 dec places
# rounding off the values upto 2 decimal places
L_test_close_new=[]
for i in range(len(y_test_acn_close_L)):
  L_test_close_new.append(round(y_test_acn_close_L[i], 2))
L_test_open_new=[]
for i in range(len(test_predict_acn_close_L)):
  L_test_open_new.append(round(test_predict_acn_close_L[i], 2))

# profit calculation
profit_calc=[]
for i in range(len(pnl_signal)):
  y1 = L_test_close_new[i] - L_test_open_new[i]
  y2 = pnl_signal[i]
  profit = y2 * y1
  if pnl_signal[i] == 1.0 :
    print(f"The signal for the day - {acn_date[i+1386]}: BUY, since profit :{profit}")
  elif pnl_signal[i] == -1.0:
    print(f"The signal for the day - {acn_date[i+1386]}: SELL, since loss :{profit}")
  profit_calc.append(profit)

zero_L=[]
for i in range(61):
  zero_L.append(0) 

# Cumulative Profit Calculation
st.subheader('Cumulative Profit for the Timeline of next 60 days')
cumu_pft= 0
for i in range(len(profit_calc)):
  cumu_pft = cumu_pft+profit_calc[i]
if cumu_pft>0:
  st.write(cumu_pft)
else :
  st.write(-cumu_pft)

# plotting the profit curve
st.subheader('Profit Curve')
fig = plt.figure(figsize = (14,6))
plt.xlabel('Date-Timeline')
plt.ylabel('Profit')
plt.plot(acn_date[1386:1447], profit_calc) 
plt.plot(acn_date[1386:1447], zero_L)
plt.legend(["Profit Curve","Neutral Line"])
st.pyplot(fig)


