# -*- coding: utf-8 -*-
"""
Elec_Forecast_collab.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import io
from google.colab import files
import warnings
warnings.filterwarnings(action='ignore')

"""### Data exploration (explore the dependent variable, analyze trends, see if additional features can be derived, format the data etc.)"""

train = pd.read_csv(r'C:\Users\abishek\Downloads\Pradeep\AV_Hackathon_TSA\train_6BJx641.csv', sep=",")
test = pd.read_csv(r'C:\Users\abishek\Downloads\Pradeep\AV_Hackathon_TSA\test_pavJagI.csv', sep=",")

train.head(5)

test.head(195)

train.info()

train['datetime']  = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in train["datetime"]]
train['Date'] = [dt.datetime.date(d) for d in train['datetime']]
train['Time'] = [dt.datetime.time(d) for d in train['datetime']]
train['Hour'] = pd.to_datetime(train['Time'], format='%H:%M:%S').dt.hour
train.head(5)

train.set_index('ID', inplace=True)

"""### Visualizing the data"""

plt.clf()
plt.figure(figsize=(15, 7))
# plt.plot(train.iloc[:10000,6].values, color = 'red')
plt.plot(train.iloc[500:600,6].values, color = 'red')
plt.title('Hourly Consumption')
plt.xlabel('Time')
plt.ylabel('Consumption')
plt.legend()
plt.show()

"""
From the above plot, we can see that there is no increasing/decreasing trend but strong seasonality is present
"""

# Descriptive stats of the all numeric variables to identify outliers
des = train.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).transpose()

train.head(5)

# Quick One Hot Encoding of var2

train['var2_A'] = train['var2'].apply(lambda x:1 if x == 'A' else 0)
train['var2_B'] = train['var2'].apply(lambda x:1 if x == 'B' else 0)
train['var2_C'] = train['var2'].apply(lambda x:1 if x == 'C' else 0)

train_new = train.drop(['datetime','var2','Time'], axis=1)
train_new.head(555)

# Since the train & test data have overlapping timelines, we have to structure the num_of_lags and the prediction window 
# in such a way that we can predict the test data (8 days) from the info from train (utmost last 23 days') data

### Trying to identify the degree of autocorrelation present in the consumption data
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(train_new.iloc[:555,4].values)

# Looks like there is strong autocorrelation till 250 lags (~10 days' data)

"""Data Pre-processing before building the model

1) Outlier Treatment - so that our model doesn't unnecessarily try to predict absurd values
2) Data Scaling - so that the loss optimization algorithm converges faster
3) Structuring the timesteps/lags as input variables into our DL model
4) Carve out a small slice from the train data for validation purposes
5) Doing all the same for the validation data & the unlabelled Test Data too
"""

def load_treat(idat, traindata=True):
    idat.set_index('ID', inplace=True)

    idat['datetime']  = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in idat["datetime"]]
    idat['Date'] = [dt.datetime.date(d) for d in idat['datetime']]
    idat['Time'] = [dt.datetime.time(d) for d in idat['datetime']]
    idat['Hour'] = pd.to_datetime(idat['Time'], format='%H:%M:%S').dt.hour

    # Descriptive stats of the all numeric variables to identify outliers
    des = idat.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).transpose()

    if traindata:
        for col in ['temperature', 'var1', 'pressure', 'windspeed', 'electricity_consumption']:
            idat[col][idat[col] > des['99%'][col]] = des['99%'][col]
            idat[col][idat[col] < des['1%'][col]]  = des['1%'][col]

    else:
        for col in ['temperature', 'var1', 'pressure', 'windspeed']:
            idat[col][idat[col] > des['99%'][col]] = des['99%'][col]
            idat[col][idat[col] < des['1%'][col]]  = des['1%'][col]

    # Quick One Hot Encoding of var2

    idat['var2_A'] = idat['var2'].apply(lambda x:1 if x == 'A' else 0)
    idat['var2_B'] = idat['var2'].apply(lambda x:1 if x == 'B' else 0)
    idat['var2_C'] = idat['var2'].apply(lambda x:1 if x == 'C' else 0)

    idat_new = idat.drop(['datetime','var2','Time'], axis=1)

    return idat_new

uploaded = files.upload()

train = pd.read_csv(io.BytesIO(uploaded['train_6BJx641.csv']))
train.head(5)

uploaded = files.upload()

test  = pd.read_csv(io.BytesIO(uploaded['test_pavJagI.csv']))
test.head(5)

train_new = load_treat(train, True)
test_new = load_treat(test, False)

train_new.describe(percentiles=[0.01,0.99])

"""
Using index 0-551 (maximum - it can be lesser too), we need to predict index 552-743
"""
consol = pd.concat((train_new, test_new), axis=0, ignore_index=False).sort_index()
consol.drop('Date', axis=1, inplace=True)
consol.head(555)

# Creating a data structure with 100 timesteps/lags (approx 4 days) and 1 output step
X = []
y = []
X_train = []
y_train = []
X_test = []

num_lags = 96
num_out = 1
test_index = list(test_new.index)
train_index = list(train_new.index)
l = len(list(consol.index))
npconsol = np.hstack((consol.iloc[:,:4].values,
                      consol.iloc[:,5:].values,
                      consol.iloc[:,4:5].values))
npconsol[:2]

# Scaling features

scaler_X = StandardScaler()
scaler_X.fit(npconsol[train_index, 0:8])

scaler_y = StandardScaler()
scaler_y.fit(npconsol[train_index, 8:9])

npconsol_scaled = np.hstack((scaler_X.transform(npconsol[:,0:8]),
                             scaler_y.transform(npconsol[:,8:9])))

npconsol_scaled[:2]

for i in range(l):
    in_end_ind = i + num_lags
    out_end_ind = in_end_ind + num_out - 1
    if out_end_ind > l:
        break
    feat, outlist = npconsol_scaled[i:in_end_ind, :-1], npconsol_scaled[in_end_ind-1:out_end_ind, -1]
    # X.append(feat)
    # y.append(outlist)
    if (out_end_ind - 1) in test_index:
        X_test.append(feat)
    else:
        X_train.append(feat)
        y_train.append(outlist)

X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
X_train.shape, y_train.shape, X_test.shape

def rmse(act,pred):
    return ((act - pred)**2).mean() ** .5

"""### Creating a Deep Learning model (with 250 lags)"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import advanced_activations

# Initialising
nn_reg = Sequential()

# 1st LSTM layer with Dropout
nn_reg.add(LSTM(units = 50, 
                activation='relu', 
                return_sequences = True, 
                input_shape = (X_train.shape[1], X_train.shape[2])))
nn_reg.add(Dropout(0.1))

# 2nd LSTM layer with Dropout
nn_reg.add(LSTM(units = 50, 
                activation='relu', 
                return_sequences = True))
nn_reg.add(Dropout(0.1))

# 3rd LSTM layer with Dropout
nn_reg.add(LSTM(
                # activation='relu',
                units = 40
                ))
nn_reg.add(Dropout(0.1))

# Output Layer
nn_reg.add(Dense(units = num_out))

from keras import backend
 
def rmse_tf(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# Compiling & Fitting
nn_reg.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics=[rmse_tf])
nn_reg.fit(X_train, y_train, epochs = 20, batch_size = 128, verbose=1)

"""# Predictions & Visualisation of results"""

# Visualising the results on train & test data

y_train_pred_sc = nn_reg.predict(X_train)
y_test_pred_sc = nn_reg.predict(X_test)

y_train_pred = scaler_y.inverse_transform(y_train_pred_sc)
y_test_pred = scaler_y.inverse_transform(y_test_pred_sc)

train_new['prediction'] = pd.Series(np.concatenate((np.zeros(num_lags-1),y_train_pred.flatten()), axis=0), index=train_new.index)
test_new['prediction'] = pd.Series(y_test_pred.flatten(), index=test_new.index)
train_new.tail(6435)

test_new.to_csv('lstm_test_pred.csv') 
files.download('lstm_test_pred.csv')

# RMSE on Train data
act_train = train_new.iloc[num_lags-1:,4].values
pred_train = train_new.iloc[num_lags-1:,10].values
train_rmse = rmse(act_train, pred_train)
train_rmse

plt.clf()
plt.plot(act_train, color = 'red', label = 'Actual Consumption')
plt.plot(pred_train, color = 'blue', label = 'Predicted Consumption')
plt.title('Hourly Consumption - Train (Act vs LSTM Fcst)')
plt.xlabel('Time')
plt.ylabel('Comsumption')
plt.legend()
plt.show()

# Saving the model & weights to virtual files

from keras.models import model_from_json
from keras.models import load_model

nn_reg_json = nn_reg.to_json()

with open("lstm_model.json", "w") as json_file:
    json_file.write(nn_reg_json)

nn_reg.save("lstm_model.h5")
nn_reg.save_weights("lstm_weights.h5")

# Downloading model from the virtual file to local machine
files.download('lstm_model.json')

# Downloading model from the virtual file to local machine
files.download('lstm_model.h5')
# Downloading weights from the virtual file to local machine
files.download('lstm_weights.h5')

"""### Compare our model with a Naive Forecasting model (value at t = value at t-1)"""

consol_nv = pd.concat((train_new, test_new), axis=0, ignore_index=False).sort_index()
consol_nv.drop('Date', axis=1, inplace=True)
consol_nv.head(555)

# consol_nv.drop('prediction', axis=1, inplace=True)

test_index = list(test_new.index)
train_index = list(train_new.index)
l = len(list(consol_nv.index))
npconsol_nv = np.hstack((consol_nv.iloc[:,:4].values,
                      consol_nv.iloc[:,5:].values,
                      consol_nv.iloc[:,4:5].values))
npconsol_nv[:2]

"""# Predictions & Visualisation of results"""

pred = [0]
for i in range(1,l):
    if np.isnan(npconsol_nv[i-1, 8]):
      pred.append(pred[-1])
    else:
      pred.append(npconsol_nv[i-1, 8])

pred = np.array(pred)
pred

np.isnan(npconsol_nv).max() , np.isnan(pred).max()

def rmse(act,pred):
    return ((act - pred)**2).mean() ** .5

consol_pred_nv = consol.copy()
consol_pred_nv['prediction'] = pd.Series(pred, index=consol.index)
consol_pred_nv.head(252)

consol_pred_nv.to_csv('consol_pred_nv.csv') 
files.download('consol_pred_nv.csv')

# train_index1 = train_index.remove(0)
act_train_nv = consol_pred_nv.iloc[train_index,4].values
pred_train_nv = consol_pred_nv.iloc[train_index,9].values

# RMSE on Train data
train_rmse_nv = rmse(act_train_nv, pred_train_nv)
train_rmse_nv

plt.clf()
plt.plot(act_ovl, color = 'red', label = 'Actual Consumption')
plt.plot(pred_ovl, color = 'blue', label = 'Predicted Consumption')
plt.title('Hourly Consumption - Train (Act vs Naive Fcst)')
plt.xlabel('Time')
plt.ylabel('Comsumption')
plt.legend()
plt.show()

"""
Thus, our DL model reduces the RMSE from ~31 (Naive forecast) to ~27. Betterment of ~4 units per hour in estimations means ~100 units per day and ~36K units per year (which is not a small amoung)
It can even be bettered through: 
1) adding more layers (changing the architecture)
2) splliting the given train set further into a true training & validation set to ensure the model generalizes well
2) more complex activation functions like ReLU or Leaky ReLU to bake in non-linearity
3) experiemnting with different architecture, activation functions, optimizers like adam with GRID searches
4) Building GRUs
"""