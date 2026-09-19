import pandas #interact with tabular data
import yfinance as yf #Yahoo's own API
import sqlite3 #SQL package
import matplotlib.pyplot as plt #Produce plots of returns by date
import numpy as np #Mathematical functiokns
from scipy.optimize import minimize #use for optimisation later on
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error,r2_score

stocks = ["GLD" , "CL=F" , "TSLA" , "LLOY.L" , "AMZN" , "NG.L" ] #Choices regarding imported stock data. Commodities. Finance and tech industry related stocks.
stockdata = yf.download(stocks, start = "2020-01-01", interval='1mo') #Downloading stock data from yahoo finance with monthly intervals
stockdata.columns = ['_'.join(col).strip() for col in stockdata.columns] #Ammending column labels to ensure column is only 1 string rather than seperate

con = sqlite3.connect('stock_portfolio.db') #Connect to sql package. Then create sep file using sqlite3 package under the database file name stock_portfolio
stockdata.to_sql('Mo_Stk_Prices',con ,if_exists='replace') #Importing stock data to file under the table name Mo_stk_prices. If_exists replaces current file during repeated runs

check=pandas.read_sql_query("Select * from Mo_stk_Prices",con) #Purely for debugging. Just to check whether data is saved.
print(check) 

#Now want to calculate monthly percentage change in closing prices to then allow for calculations of potential return
all_returns = pandas.DataFrame() #Useful later on when performing optimisation. Will help store stock and its equivalent return


for s in stocks: # for loop based on stocks initially chosen.
    stockclose = f'Close_{s}' #defining variable s and stockclose
    mp_query = f'''
        SELECT Date,
               (("{stockclose}" - LAG("{stockclose}") OVER (ORDER BY Date)) / LAG("{stockclose}") OVER (ORDER BY Date)) * 100 AS "Percentage_{s}_return"
        FROM Mo_Stk_Prices
    ''' #SQL query. Exploiting data from Mo_stk_prices table to output date and % return for each stock
    monthly_returns = pandas.read_sql_query(mp_query, con)
    monthly_returns=monthly_returns.fillna(0) #Fill NA values within the percentage return with 0

    pound=1
    cum_return=[] #define a list for the cumulative return
    for r in monthly_returns[f"Percentage_{s}_return"]: #Creating a for loop to calc return for each stock
            pound = pound*(1+r/100) 
            cum_return.append(pound)   #Adding new value to the list using append()     

    plt.plot(monthly_returns["Date"] , cum_return, label = s) #plotting date against the cumulate return
    plt.legend()

    #Now want to produce risk metrics 
    #Volatility, Sharpe ratio and Value at Risk - All metrics that can help deciding whether investment into stock is suitable
    monthly_vol = monthly_returns[f'Percentage_{s}_return'].std()
    annual_vol = monthly_vol* np.sqrt(12)  #convert by multiplying monthly volatility with square root of 12    
    av_ret = monthly_returns[f"Percentage_{s}_return"].mean() / 100   
    rfr = 0.0391 / 12 #
    sharpe = (av_ret -rfr)  /(monthly_vol/100)#Calculating the sharpe ratio
    VaR = np.percentile(monthly_returns[f"Percentage_{s}_return"] , 5)

    Metrics = pandas.DataFrame({
         'Stock name' : [s] ,
         'Annual volatility' : [annual_vol],
         'Risk free rate' : [rfr],
         'Sharpe ratio' : [sharpe] ,
         'Value at risk' : [VaR]
    })
    print(Metrics) #Producing table of calculated parameters
    
    if all_returns.empty: #Using an if loop to fill the all_returns variable. 
        all_returns= monthly_returns
    else:
         all_returns = all_returns.merge(monthly_returns, on = 'Date')

con.close()

plt.xlabel("Date")
plt.xticks(rotation = -45 , fontsize=2) #shrinking values on x axis
plt.ylabel("Return per £")
plt.title("A plot of Return per pound against date")

#Now looking to perform Markowitz optimisation - i.e Helps to determine the best method of approach to investment in the stock market to produce the greatest return.
#Found this when looking at ways to actually quantify whether a stock is worth investing into
#Need sharpe ratio for determined stock proportions

cov_mtx = all_returns.drop(columns = 'Date') . cov()    #First have to determine covariance matrix. This determines the strength of relationships between stocks  
print(cov_mtx)

#Now want mean return per stock
return_columns = [f'Percentage_{s}_return' for s in stocks] #storing returns for each stocks as an array using a nested for loop. 
mean_stock_return = all_returns[return_columns].mean()
print(mean_stock_return)

#Now using the covariance matrix and the mean, have to determine whether the performance of the new stock portfolio using corresponding values from the 

def port_performance(weights,mean_stock_return, cov_mtx,rfr): #Defining the function port_performance and the parameters involved
     portfolio_return = np.sum(weights*mean_stock_return) #Calculating return based on the average return and the weighting
     portfolio_vol=np.sqrt(np.dot(weights,np.dot(cov_mtx,weights))) #Determining how assets move together and their volatility. NOTE this bit took a very long time so 2x check
     sharpe = (portfolio_return - rfr)/ (portfolio_vol) #Calculating the sharpe ratio
     return portfolio_return, portfolio_vol,sharpe #Stores metrics

def negative_sharpe(weights, mean_stock_return, cov_mtx, rfr):  #defined a function to return a negative sharpe value
    portfolio_return,portfolio_vol,sharpe = port_performance(weights, mean_stock_return,cov_mtx,rfr)
    return -sharpe 

def portfolio_volatility(weights, mean_stock_return, cov_mtx):
    return np.sqrt(np.dot(weights, np.dot(cov_mtx, weights)))
                                                                            
num_stocks = len(stocks) #States how many stocks are within the stock list
constraint = ({'type':'eq', 'fun' :lambda w:np.sum(w) - 1})#Here i am setting a constraint up. This is ensuring the weights of stocks = 1. Lambda is used to define a small function without using def.
#type: eq is for an equality constraint. 

bound = tuple((0,1) for _ in range(num_stocks)) # creating a bound for each stock's weight. Found out I can use _ when using a for loop when i dont have to loop a set num of times. Then will produce bounds for each value in the stock

#now want an initial guess for the weightings of each
#want a list of equal weightings. So 1/(number of stocks)

first_guess = num_stocks*[1./num_stocks] #Creating a variable for my first guess. 

result = minimize(negative_sharpe, first_guess, args=(mean_stock_return,cov_mtx,rfr), method ='SLSQP', bounds = bound, constraints=constraint)
optimal_weights=result.x

opt_table = pandas.DataFrame({ #Produce a table of the optimal weights for each stock 
     'Stock' :stocks,
    'Optimal weight' : optimal_weights
})
print(opt_table)

target_returns = np.linspace(mean_stock_return.min(), mean_stock_return.max(), 50)
frontier_volatility = []

for target in target_returns:
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, target=target: np.sum(w * mean_stock_return) - target}
    )
    result = minimize(portfolio_volatility, first_guess, args=(mean_stock_return, cov_mtx),
                       method='SLSQP', bounds=bound, constraints=constraints)
    frontier_volatility.append(result.fun)

# 3. Plot the frontier, plus your optimal and equal-weighted portfolios
plt.figure(figsize=(10, 6))
plt.plot(frontier_volatility, target_returns, label='Efficient Frontier')

opt_return, opt_vol, opt_sharpe = port_performance(optimal_weights, mean_stock_return, cov_mtx, rfr)
plt.scatter(opt_vol, opt_return, color='red', marker='*', s=200, label='Max Sharpe Portfolio')

equal_weights = np.array(num_stocks * [1. / num_stocks])
eq_return, eq_vol, eq_sharpe = port_performance(equal_weights, mean_stock_return, cov_mtx, rfr)
plt.scatter(eq_vol, eq_return, color='blue', marker='o', s=100, label='Equal-Weighted Portfolio')

plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')

x = all_returns[[f'Percentage_{s}_return' for s in stocks]].shift(1).dropna()
y=all_returns[f'Percentage_GLD_return'].iloc[1:]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2 , shuffle=False)
model=RandomForestRegressor() #Using random forest regression model
model.fit(x_train,y_train) 

prediction=model.predict(x_test)
mse = mean_squared_error(y_test,prediction)
r2=r2_score(y_test,prediction)

print(f'Mean squared error is {mse} , R^2 is {r2}') #Printing mean squared error and r^2

comparison = pandas.DataFrame({#Table of test values and predicted outputs
     'Actual':y_test.values,
     'Predicted' :prediction
})
print(comparison)

#Now want to validate whether optimised portfolio was reasonable

optimised_port_return = sum(all_returns[f'Percentage_{s}_return'] * w for s, w in zip(stocks , optimal_weights))
optimised_cum_return = [] #create array
pound=1
for r in optimised_port_return:
    pound=pound*(1+r/100)
    optimised_cum_return.append(pound)

equal_port_return = sum(all_returns[f'Percentage_{s}_return']*w for s, w in zip(stocks,equal_weights))

equal_cum_return=[] #creating an array for the cumulative return when equal weights are used
pound=1
for r in equal_port_return:
    pound=pound*(1+r/100)
    equal_cum_return.append(pound)

plt.figure(figsize=(10, 6))
plt.plot(all_returns['Date'], optimised_cum_return, label='Optimized Portfolio')
plt.plot(all_returns['Date'], equal_cum_return, label='Equal-Weighted Portfolio')
plt.xlabel('Date')
plt.ylabel('Growth of £1')
plt.title('Backtest: Optimized vs Equal-Weighted Portfolio')
plt.legend()
plt.show()

print(f"Optimized portfolio final value: £{optimised_cum_return[-1]:.2f}")
print(f"Equal-weighted portfolio final value: £{equal_cum_return[-1]:.2f}")
print(f"Optimized — Return: {opt_return:.2f}%, Volatility: {opt_vol:.2f}%, Sharpe: {opt_sharpe:.2f}")
print(f"Equal-weighted — Return: {eq_return:.2f}%, Volatility: {eq_vol:.2f}%, Sharpe: {eq_sharpe:.2f}")