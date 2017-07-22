import datetime as dt
import random
import re
import pandas as pd
import quandl 
import numpy as np
import StockPredictor


##########Parameters##########################
#Used for training start and end
start_date = '2016-01-03'
end_date = '2016-12-30'

# Used for predicting the value on the query date
query_date = '2017-01-05'

# Trains on one of these stocks. If more than one are given, picks a random stock.
#stocks_query = ['WIKI/BAC','WIKI/JPM', 'WIKI/ZION', 'WIKI/CFG', 'WIKI/MS']
stocks_query =['WIKI/BAC']

##########################################################

metric = 'Adj. Close'
required_metric = ' - Adj. Close'


#Quandl API key
api_key = 'SREH9ESaPvQ86-oeTRQG'
quandl.ApiConfig.api_key = api_key

# Avoid fetching from quandl
fetch_remote = True

#Select stocks which are closely related so the model would fit across these stocks.
filename = 'clustered_stocks_data.csv'

if fetch_remote:
    print "Fetching from Quandl"
    merged_dataset = quandl.MergedDataset(stocks_query)

    data_all = merged_dataset.data(
        params={'start_date': start_date, 'end_date': end_date}).to_pandas()

    data = pd.DataFrame(index=data_all.index)
    for q in stocks_query:
        data[q] = data_all[q + " - " + metric]

    # Write to CSV to avoid fetching from quandl
    data.to_csv(filename)
    # Loading to set correct index col.
    data = pd.read_csv(filename,parse_dates=[0], index_col='Date')
else:
    print "Fetching from csv"
    data = pd.read_csv(filename,parse_dates=[0], index_col='Date')

# Calculate percentage change from one day to the next day
data = data.pct_change()

#Forward and backfll blank data
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

monthly = data.asfreq('BM', method='ffill')

monthly_return = monthly.pct_change()
monthly_return.fillna(method='bfill', inplace=True)
#print(monthly_return.var())

# Transpose the describe result to do calculation below
res = data.describe().transpose()
res['variance'] = data.var()
#print(res) 
# Pick a random stock
#tickr = random.sample(res.transpose().keys(), 1)[0]
df = pd.DataFrame()
for tickr in stocks_query:
    print "Selected stock for prediction: ", tickr

    sp = StockPredictor.StockPredictor(api_key,query_date)

    # Fetch data from Quandl API 
    sp.fetch_data(tickr, start_date, end_date, metric, fetch_remote=fetch_remote, filename= "ticker_{}.csv".format(re.sub(r'([^\s\w]|_)+', '', tickr)))
    actual_price, end_date_price = sp.do_testing()

    # Preprocessing 
    sp.preprocessing(  seq_length=5)

    # Implementation for Linear Regression
    sp.trainLR()
    predicted_price = sp.make_predict(data)
    df1 = sp.print_results(actual_price, predicted_price, end_date_price)
    
    # Implementation for SVR 
    sp.trainSVR()
    predicted_price = sp.make_predict(data)
    df2 = sp.print_results(actual_price, predicted_price, end_date_price)
    df1.append(df2)
    # Implementation for KNN
    sp.trainKNN() 
    predicted_price = sp.make_predict(data)
    df3 = sp.print_results(actual_price, predicted_price, end_date_price)
    df1.append(df3)
    
    df = pd.concat([df,df1],axis = 1)
df.to_csv('result_metrics.csv')
print("###############################################")

# Visualize bollinder bands
[sp.plot_bollinger(data, tkr) for tkr in data.columns]
