# MONTE CARLO PORTFOLIO SIMULATION
#--------------------------------#

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

# define random seed for numpy
np.random.seed(42)

# define function to retrieve data using yfinance
def get_data(stocks, start, end):
    
    # use yahoo api to get data
    stock_data = yf.download(stocks, start=start, end=end)
    
    # extract closing price
    stock_data = stock_data['Close']
    
    # extract returns
    # use log returns
    returns = np.log(stock_data) - np.log(stock_data.shift(1))
    
    # extract mean returns
    mean_returns = returns.mean()
    
    # create covariance matrix
    cov_matrix = returns.cov()
    
    #return the mean returns and cov matrix
    return mean_returns, cov_matrix

# create portfolio of stocks
#--------------------------#

# here I am using a collection of 10 technology stocks listed on the NASDAQ
# for yahoo finance to get the data, the tickers need to be submitted as a list

# create list of stocks
stocks = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'TSM', 'TSLA', 'NVDA', 'INTC', 'ADBE', 'NFLX']

# set the end date to today
end_date = dt.datetime.now()

# set start date to X years back from today
years = 10
# set start date
start_date = end_date - dt.timedelta(days = (years * 365))

# run function to get mean_returns and cov_matrix
mean_returns, cov_matrix = get_data(stocks, start_date, end_date)

# BUILD MONTE CARLO SIMULATION MODEL
#----------------------------------#

# define number of sims
sims = 1000

# create timeframe for projection
t = 252

# create numpy array with specific shape
# the t input defines the number of rows
# len(weights) defines the number of columns - one for each stock
# the individual names of the stocks are not relevant, only the mean return
mean_matrix = np.full(shape = (t, len(stocks)), fill_value = mean_returns)

# transpose the array
# now we have an array with a row for each stock and mean return for every day (t) as columns
mean_matrix = mean_matrix.T

# create a new array to store the results of the simulation
# initiate the values as zero
# can initialize with only 0, but 0.0 is more concise since we are working with floats
portfolio_sims = np.full(shape = (t + 1, sims), fill_value = 0.0)

# define risk free rate for sharpe ratio
# as of writing we can say about 4% since looking at US stocks
risk_free_rate = 0.04

# set initial portfolio value
initial_portfolio = 10000

# create list to store the weights of each sim
sim_results = []
# loop and run simulations
for i in range(sims):
    
    # generate random weights for each stock in the portfolio
    weights = np.random.random(len(stocks))

    # normalize the weights so the weights are never more than 1
    # this function divides each element in the weights list by the sum
    # this modifies the original array and sums to 1
    weights /= np.sum(weights)

    # calculate correlated daily returns
    daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, t).T
    # calculate portfolio return for each sim
    # multiply weights and daily returns from matrices
    weighted_simulated_returns_raw = np.dot(daily_returns.T, weights)
    # get the annual std
    annual_std = np.std(weighted_simulated_returns_raw) * np.sqrt(252)
    # get annual return of portfolio
    annual_compounded_return = np.exp(np.mean(weighted_simulated_returns_raw) * 252) - 1
    # calculate the annual sharpe ratio
    sharpe_ratio = (annual_compounded_return - risk_free_rate) / annual_std
    # append the results to list
    sim_results.append((weights, annual_compounded_return, annual_std, sharpe_ratio))
    
    # simulate returns for monte carlo plot
    portfolio_value = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_portfolio
    # assign the value to the correct index of array
    # also insert initial portfolio value at index 0 for clarity
    portfolio_sims[:, i] = np.insert(portfolio_value, 0, initial_portfolio)

# create plot of results
plt.plot(portfolio_sims)
plt.xlabel('Days')
plt.xlim(0, 252)
plt.ylabel('Portfolio Value')
plt.title('Monte Carlo Simulation')
plt.show()

# EXTRACT THE MAX SHARPE FROM SIMULATIONS
#---------------------------------------#

# then recalculate metrics based on historical data
# now have more realistic return estimation for the monte carlo port

# get values from simulation list
#------------------------------------------------------------------#
# extract the mean return from all portfolios
mean_return = np.mean([result[1] for result in sim_results])
mean_std = np.mean([result[2] for result in sim_results])
mean_sharpe = np.mean([result[3] for result in sim_results])

# get max sharpe ratio
max_sharpe = max(sim_results, key=lambda x: x[3])[3]
# get weights for max sharpe
max_sharpe_weights = max(sim_results, key=lambda x: x[3])[0]
max_sharpe_weights = [round(weight, 2) for weight in max_sharpe_weights]
# extract max_sharpe return
max_sharpe_simulation_return = max(sim_results, key=lambda x: x[3])[1]
# get simulated annual std for max_sharpe port
max_sharpe_std = max(sim_results, key=lambda x: x[3])[2]

# rerun calculations to get more realistic stats
#----------------------------------------------#

# get expected return for max sharpe weights 
exp_port_return_daily = np.dot(mean_returns, max_sharpe_weights)
# scale to annual
hist_max_sharpe_ann_return = np.exp(exp_port_return_daily * 252) - 1
# get hist std
hist_max_sharpe_std = np.sqrt(np.dot(np.array(max_sharpe_weights).T, np.dot(cov_matrix, max_sharpe_weights))) * np.sqrt(252)
# calc new sharpe ratio
adj_sharpe_ratio = (hist_max_sharpe_ann_return - risk_free_rate) / hist_max_sharpe_std
# create dataframe of the stocks and the weights
stocks_weights = pd.DataFrame({'Asset': stocks, 'Weights': max_sharpe_weights})
stocks_weights = stocks_weights.sort_values(by='Weights', ascending=False)

# PLOT THE RESULTS
#----------------#

# import color map for bar chart
from matplotlib import cm
# create color map
color_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# reverse for descending gradient
rev_colors = color_list[::-1]
color = cm.Reds(rev_colors)

# plot the weights
plt.figure(figsize=(10, 8))
plt.bar(stocks_weights['Asset'], stocks_weights['Weights'], color=color, edgecolor='black')
plt.xlabel('Asset', fontdict={'fontsize': 14})
plt.ylabel('Weight', fontdict={'fontsize': 14})
plt.title('Max Sharpe Portfolio for Monte Carlo Sim.', fontdict={'fontsize': 16})
plt.show()

