import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

company = 'FB'
start = dt.datetime(2012,1,1)
end = dt.datetime(2022,12,1)

data = web.DataReader(company, 'yahoo', start,end)
print(data.head(2))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
prediction_days = 60

X_train = []
y_train = []

for x in range(prediction_days,len(scaled_data)):
    X_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1))

model = Sequential

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit(X_train,y_train,epochs=25,batch_size=32)
model.save('C:/Users/Jagos/Documents/GitHub/Big-and-Small-ML-Projects/Stock Price Prediction(LSTM)/model.h5')

