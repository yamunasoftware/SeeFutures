### MODEL IMPORTS ###

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

### MODEL FUNCTIONS ###

# Predict Price:
def predict_price(recent_prices):
  # Fits the Pricing Data:
  model, x_train, x_test, y_train, y_test, scaler = build_model()
  fitted_prices = scaler.transform(np.array(recent_prices).reshape(-1, 1))
  fitted_prices = fitted_prices.reshape((1, len(fitted_prices), 1))

  # Predicts the Price:
  prediction = model.predict(fitted_prices)
  return scaler.inverse_transform(prediction)[0][0]

# Visualize Pricing:
def visualize_pricing():
  # Builds Model and Runs Prediction:
  model, x_train, x_test, y_train, y_test, scaler = build_model()
  predictions = model.predict(x_test)

  # De-Normalizes Data:
  predictions = scaler.inverse_transform(predictions)
  prices = scaler.inverse_transform(y_test.reshape(-1, 1))

  # Plots Predictions and Prices:
  plt.figure(figsize=(10, 5))
  plt.plot(predictions, label='Prediction')
  plt.plot(prices, label='Price')
  plt.xlabel('Time')
  plt.ylabel('Price')
  plt.legend()
  plt.show()

# Evaluate Model:
def evaluate_model():
  # Builds Model and Evaluates:
  model, x_train, x_test, y_train, y_test, scaler = build_model()
  train_loss = model.evaluate(x_train, y_train)
  test_loss = model.evaluate(x_test, y_test)

  # Outputs Evaluation Values:
  print('Training Loss: ' + str(train_loss))
  print('Test Loss: ' + str(test_loss))

# Builds the LSTM Model:
def build_model():
  # Imports and Normalizes:
  dataframe = pd.read_csv('dataset.csv', parse_dates=['date'], index_col='date')
  scaler = MinMaxScaler()
  dataframe['price'] = scaler.fit_transform(dataframe[['price']])

  # Feature Lag (10 Days of Data):
  lag = 10

  # Data Values:
  data = dataframe['price'].values
  i = 0

  # Gets the Lagged Features:
  x, y = [], []
  while i < len(data) - lag:
    x.append(slice_lag_features(data, i, (i + lag)))
    y.append(data[i + lag])
    i += 1

  # Splits Data Into Train and Test Data:
  x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.1, shuffle=False)

  # Reshape Data for Time Series Model:
  x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
  x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

  # Builds the Prediction Model:
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(lag, 1)))
  model.add(LSTM(50))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Trains the Prediction Model:
  model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), verbose=0)
  return model, x_train, x_test, y_train, y_test, scaler

# Slice Lag Features:
def slice_lag_features(data, start, end):
  # Array Values:
  array = []
  i = start

  # Slice Array:
  while i < end:
    array.append(data[i])
    i += 1
  return array