# Forecasting Hourly Electricity Prices
  Various models and approaches are deployed to forecast hourly electricity prices provided by [the EXIST Market Transparency Platform](https://seffaflik.epias.com.tr/transparency/). First, lag-1 of the exogenous variable are used to make prediction with values at time t-1 to forecast the electricity price at time t (one hour later). For that, a baseline XGBM deployed to observe what pre-fine-tuning performance looks like. Afterwards, XGBM model tuned with various hyperparameters including but not limited to max_depth, n_estimators, subsamples and so on. Fine-tuned XGBM model revealed better performance in terms of root mean squared errors than what the baseline model achieved. 
  
  Finally, nn.LSTM model developed to use the capabilities of LSTM algorithm with long-term time-dependencies. Nn.LSTM Time-series model achieved better (less rmse) performance compared to fine-tuned XGBM model. Below you can find the model architecture. 48-hour back-horizon is used to predict horizon of +1.  
  
## Data Description
| Variable Name | Description |
|---------------|-------------|
| PTF | Market Clearing Price is the hourly energy price that is determined with respect to orders that are cleared according to total supply and demand| 
| Volume| In the Day Ahead Market, it is the hourly total financial value of the matching bids. (Matching bid = Matching Offer)|
| bid_amount| Sum of hourly, block and flexible bid order volumes at 0 TL/MWh price step.|
| ask_amount| Sum of hourly, block and flexible sales order volumes at 4800 TL/MWh price step.|
| imbalance_delta| Imbalance Quantity|
  
## Model Architecture  
![image](https://github.com/dfavenfre/electricity-price-forecasting/assets/118773869/9ee3babe-8637-4a73-809b-5a1ec91d2e6f) 


## XGBM Prediction Performance
![image](https://github.com/dfavenfre/electricity-price-forecasting/assets/118773869/f7c60b0f-54ae-48d7-9473-1b65f298b164)


## nn.LSTM Prediction Performance
![image](https://github.com/dfavenfre/electricity-price-forecasting/assets/118773869/ce3a63d3-5cf5-4998-8a1f-591c6bdd37aa)



## Model Performances

|Model|	RMSE|
|-----|-----|
|nn.LSTM|	57.348|
|Fine-tuned XGBM|	59.972|
|Baseline XGBM|61.42|

