# import packages
import pandas as pd
import numpy as np
import fredapi as fa
import yfinance as yf

# GET S&P500 DATA
#---------------#

# set ticker for sp500
ticker_yf = "^GSPC"

# set start date to match other data
start_date_yf = '1900-01-01'

# set end date
end_date_yf = '2023-10-01'

# set interval to monthly
interval = "1mo"

# get stock market return data
market_data = yf.download(ticker_yf, start=start_date_yf, end=end_date_yf, interval=interval)

# create df to extract closing prices
market_data = pd.DataFrame(market_data['Close'].astype('float'))

# calculate log returns for each day using numpy log method
# shift function returns the close in the previous row
market_data['log_returns_raw'] = np.log(market_data['Close'] / market_data['Close'].shift(1))
# drop the close column
market_data.drop(columns=['Close'], inplace=True)
# drop the single NaN from the first row
market_data = market_data.dropna()

# GET ECONOMIC DATA FROM FRED
#---------------------------#

# will use S&P500 as target so will focus on US data
# could be done for any other country as well with same factors

# enter api key
# to get an api key, simply go to FRED website an make account
fred_api_key = 'ENTER KEY HERE'

# set up access
fred = fa.Fred(api_key=fred_api_key)

# get unemployment rate
unrate = fred.get_series('UNRATE')
# convert unemployment rate to percentage
unrate /= 100

# 10 year rate data
treasury_10_yr = fred.get_series('GS10')
# convert data to percentage yield
treasury_10_yr /= 100

# get full ppi data
ppi = fred.get_series('PPIACO')
# convert to delta to make more sensible in analysis
delta_ppi = ppi.pct_change()

# get housing data
# measure of home prices in the us
case_shiller_idx = fred.get_series('CSUSHPINSA')
# calculate the delta for housing
case_shiller_idx_delta = case_shiller_idx.pct_change()

# get retail sales data
retail_sales = fred.get_series('RSAFS')
# convert to growth rate
delta_retail_sales = retail_sales.pct_change()

# get corporate earnings data
corporate_earnings = fred.get_series('CP')
# convert to change
# this data is quarterly
# maybe not worth including
# would be hard to incorporate
corp_earnings_delta = corporate_earnings.pct_change()

# incorporate ome sort of sentiment var
consumer_confidence = fred.get_series('UMCSENT')
# drop prior to 1977-12-01 00:00:00 since tons of NaN values
consumer_confidence = consumer_confidence[consumer_confidence.index > '1977-12-01 00:00:00']
# convert to % of max so it fits better with other vars
consumer_confidence /= max(consumer_confidence)

# get inflation data
cpi = fred.get_series('CPIAUCNS')
# convert to delta
cpi_delta = cpi.pct_change()

# get leading economic indicator
# actually don't include this since much of the information in it is already captured
lei = fred.get_series('USSLIND')

# create a list of tuples with the series and the name for col
economic_indicators_list = [
    ('unrate', unrate),
    ('treasury_10_yr', treasury_10_yr),
    ('delta_ppi', delta_ppi),
    ('case_shiller_idx_delta', case_shiller_idx_delta),
    ('delta_retail_sales', delta_retail_sales),
    ('consumer_confidence', consumer_confidence),
    ('cpi_delta', cpi_delta),
    ('sp500_log_returns', market_data['log_returns_raw']) # since this is df we need to extract as series
]

# concat into full dataset
# concat operation consists of two list comprehensions
# first extracts the series from our tuples list
# second extracts the keys to use as column names
economic_indicators = pd.concat(
    [indicator[1] for indicator in economic_indicators_list],
    axis=1, 
    keys=[indicator[0] for indicator in economic_indicators_list]
)
# shift the return values back by one
economic_indicators['sp500_log_returns'] = economic_indicators['sp500_log_returns'].shift(1)
# drop nan values
economic_indicators = economic_indicators.dropna()

# SET UP MACHINE LEARNING MODEL TO PREDICT RETURNS
#------------------------------------------------#

# since data set is relatively small, we should use random forest
# we can use techniques such as cross validation to reduce the likelihood of overfitting
# the main goal is to tune the hyperparameters and set up a model that is least prone to overfitting

# import packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate, KFold

# independent variables
ind_vars = economic_indicators.drop(columns=['sp500_log_returns'])

# get target variable
target_var = economic_indicators['sp500_log_returns']

# set up variables
X_train, X_test, y_train, y_test = train_test_split(ind_vars, target_var, test_size=0.3, random_state=42)

# set up the model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, max_features='sqrt', random_state=42)

# fit the model
rf_model.fit(X_train, y_train)

# set up cross validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# create scoring list to extract metrics
scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']

# perform cross validation
cv_results = cross_validate(rf_model, ind_vars, target_var, cv=kf, scoring=scoring)

# extract the scores
mse_scores = -cv_results['test_neg_mean_squared_error']
mae_scores = -cv_results['test_neg_mean_absolute_error']

# get the average and std of each
mse_avg_std = {'mse_std': np.std(mse_scores), 'mse_avg': np.mean(mse_scores)}
mae_avg_std = {'mae_std': np.std(mae_scores), 'mae_avg': np.mean(mae_scores)}

# plot actual vs predicted values and get metrics
#-----------------------------------------------#

# import the packages
import matplotlib.pyplot as plt

# extract the predicted values
y_predicted = rf_model.predict(ind_vars)

# get the feature importances
feature_importances = rf_model.feature_importances_

# combine feature score and feature name
features_list = [(name, feature) for name, feature in zip(ind_vars, feature_importances)]

# sort list in ascending order
# we use anonymous 
sorted_features = sorted(features_list, key=lambda x: x[1])
print(sorted_features)

# get the r2 score
r_squared = r2_score(target_var, y_predicted)
print(f'r_squared: {r_squared}')

# create scatter plot of predicted and actual values
plt.figure(figsize=(8, 6))
plt.scatter(target_var, y_predicted, marker='o', color='purple')
plt.xlabel('actual monthly return')
plt.ylabel('predicted monthly return')
plt.title('predicted vs. actual returns for random forest model', fontsize=16)
plt.grid(True)

# adjust the ticks so there are common steps
# get min and max values to set axis limits
min_val = min(min(target_var), min(y_predicted ))
max_val = max(max(target_var), max(y_predicted))

# set the step amount to 0.05
step = 0.05

# set the limits to max and min =/- step for some breathing room
plt.xlim(min_val - step, max_val + step)
plt.ylim(min_val - step, max_val + step)

# create the tick marks
# nest list comprehension inside ticks
# identify range by setting min and max step increments in range method
# tells us how many tick marks to make
# add 2 so final value isn't excluded and plot has breathing room
plt.xticks([i * step for i in range(int(min_val / step) - 1, int(max_val / step) + 2)])
plt.yticks([i * step for i in range(int(min_val / step) - 1, int(max_val / step) + 2)])

plt.show()