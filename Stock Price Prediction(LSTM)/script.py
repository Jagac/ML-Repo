import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
from keras.layers import *
from keras.models import *
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit


df=pd.read_csv('MSFT.csv',na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
print(df.shape)

output_var = pd.DataFrame(df['Adj Close'])
features = ['Open', 'High', 'Low', 'Volume']


scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)




