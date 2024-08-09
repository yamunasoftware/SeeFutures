# SeeFutures

Gold Commodity Futures Predictions Model

## Information

The purpose of this Python application is to allow for simple input into a model so that common people can access predictive modeling for gold commodity pricing. This model uses an LSTM (Long Short Term Memory) time-series model, which is fitted to the training dataset of past Gold price trends. The time-series model uses lagged features, which means it takes into account the last 10 days of pricing data when making a prediction. In the context of training, this means that it uses the last 10 days of pricing data to form the pricing trend, to which the model fits. The same is used in the application, where the user has to input the last 10 days of pricing data and the application will output a plot of the price and the predicted price on the plot, as well as in the terminal.

## Usage

To use the application, simple use the Bash or Batch files listed in the main directory. If you use Windows, run the batch script by running ```see_futures.bat``` in the command prompt. If you use MacOS, Linux, or GitBash, run the following command in the terminal: ```bash see_futures.sh```.

## Dependencies

- pandas
- matplotlib
- numpy
- tensorflow
- keras
- sklearn

To install these dependencies simply run the pip commands listed below.

```
pip install pandas
pip install matplotlib
pip install numpy
pip install tensorflow
pip install keras
pip install scikit-learn
```