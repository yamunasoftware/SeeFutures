### MAIN IMPORTS ###

import model
import pandas as pd
import matplotlib.pyplot as plt

### MAIN SCRIPT RUNNING ###

# Array Variables:
recent_prices = []
i = 0

# Loops through Lagged Days:
while i < 10:
  next_price = input('Enter Next Price: ')
  recent_prices.append(float(next_price))
  i += 1

# Gets the Output of the Model:
prediction = model.predict_price(recent_prices)
print('\nPredicted Price: ' + str(prediction))

# Sets the Dataframe:
dataframe = pd.DataFrame({
  'Time': range(10),
  'Price': recent_prices
})

# Plots the Prediction:
dataframe.plot(x='Time', y='Price', kind='line')
plt.plot(10, prediction, 'ro')
plt.savefig('prices.png', bbox_inches='tight')
plt.close()