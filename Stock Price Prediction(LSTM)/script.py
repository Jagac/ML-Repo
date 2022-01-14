import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
from keras.layers import *
from keras.models import *
import keras.backend as K

df=pd.read_csv('MSFT.csv',na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
print(df.shape)


