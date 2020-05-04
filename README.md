Electricity Consumption Forecasting:

An electricity distribution company wants to accurately predict demand to do better capacity planning. 
Hourly electricity consumption data for the company is given for a period of ~5 years.
The first 23 days of every month is split as Train data and the rest is split as Test Data. 
Exogenous features like temperature, windspeed, pressue etc are given in addition to the electricity consumption at every hourly point.
We are building a simple forecasting solution consisting of a Naive Forecast as benchmark and a DL model (LSTM with a basic architecture) as our model.
The labels are not present in the Test data since this was a hackathon. So, we've trained the model on our entire training data and only added dropout regularization to minimize overfitting.
This is not the ideal way to train our model. Better alternatives would ne splliting the given train set further into a true training & validation set to ensure the model generalizes well.


"""
Results:

Our DL model reduces the RMSE from ~31 (Naive forecast) to ~27. Betterment of ~4 units per hour in estimations means ~100 units per day and ~36K units per year (which is not a small amoung)
It can even be bettered through: 
1) adding more layers (changing the architecture)
2) more complex activation functions like ReLU or Leaky ReLU to bake in non-linearity
3) experiemnting with different architecture, activation functions, optimizers like adam with GRID searches
4) Building GRUs instead of LSTMs
"""
