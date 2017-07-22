from sklearn import svm, metrics, preprocessing
import datetime as dt
import pandas as pd
import numpy as np
import quandl
from tabulate import tabulate
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
import re
from sklearn.metrics import r2_score
import matplotlib.pyplot as plotter
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
#from sklearn.learning_curve import learning_curve

class StockPredictor:

    def __init__(self, api_key, query_date):
        quandl.ApiConfig.api_key = api_key
        self.ticker = ''
        self.query_date = query_date
        

    #fetch_data call queries the API, caching if necessary
    def fetch_data(self, ticker, start_date, end_date, metric, fetch_remote=False, filename='ticker.csv'):

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.metric = metric
        self.fetch_remote = fetch_remote
        self.df_metrics = pd.DataFrame(columns=[ticker])
        if fetch_remote:
            data = quandl.Dataset(ticker).data(
                params={'start_date': start_date, 'end_date': end_date}).to_pandas()

            # save as CSV 
            data.to_csv(filename)

            data = pd.read_csv(filename)
        else:
            data = pd.read_csv(filename)
        data.fillna(method='ffill', inplace=True)
        self.data = data

    def preprocessing(self, seq_length=5):
        """
        To create a sequence of days as features, this will be used for training dataset.
        """
        # number of days to predict forward
        self.query_date = dt.datetime.strptime(self.query_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(self.end_date, "%Y-%m-%d")

        # Work day counts to create uniform sequences
        self.forecast_days = abs(np.busday_count(self.query_date, end_date))
        print("Market days(forecast): {} ".format( self.forecast_days))

        self.seq_length = seq_length
        
        data = self.data
        
        # Calculate date delta
        data['Date'] = pd.to_datetime(data['Date'])
        data['date_diff'] = (data['Date'] - data['Date'].min()) / np.timedelta64(1, 'D')
        #create the sequence df
        ts_data = pd.DataFrame(index=data.index)

        # shift to get the price x days ahead, and then transpose
        for i in xrange(0, seq_length + self.forecast_days):
            ts_data["ts_data%s" % str(i + 1)] = data[self.metric].shift(1 - i)
            
        ts_data.shift(-2)
        ts_data['date_diff'] = data['date_diff']
        # Get x days later to create sequence data
        date_diffs = ['date_diff']
        for i in xrange(0, seq_length):
            date_diffs.append("ts_data%s" % str(i + 1))
        col_name = 'ts_data' + str(seq_length + self.forecast_days)

        rowcalcs = ts_data[date_diffs].dropna()
        
        self.final_row_unscaled = rowcalcs.tail(1)
        self.unscaled_data = rowcalcs
        
        ts_data.dropna(inplace=True)
        #print("Col name:")
        
        label = ts_data[col_name]
        #print(label)
        new_data = ts_data[date_diffs]
        
        #scale the data
        self.scaler = preprocessing.StandardScaler().fit(new_data)
        scaled_data = pd.DataFrame(self.scaler.transform(new_data))

        self.scaled_data = scaled_data
        self.label = label
        
    def ts_train_test_split(self):
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(self.scaled_data):
            X_train, X_test = self.scaled_data.loc[self.scaled_data.index[train_index]], self.scaled_data.loc[self.scaled_data.index[test_index]] 
            y_train, y_test = self.label.loc[self.label.index[train_index]], self.label.loc[self.label.index[test_index]]  
        #print(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test
  
    #Linear Regression trainer
    def trainLR(self):
        
        regr = LinearRegression()
        model_name = type(regr).__name__
        print("############{}################".format(model_name))
        #X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label, test_size=0.25,
        #                                                    random_state=41)
        #X_train, X_test, y_train, y_test = TimeSeriesSplit(n_splits=3)
        X_train, X_test, y_train, y_test = self.ts_train_test_split()
        
  
        parameters = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
        r2_scorer = metrics.make_scorer(metrics.r2_score)
        grid = GridSearchCV(regr, parameters, scoring=r2_scorer)
        grid.fit(X_train, y_train)
        predict_train = grid.predict(X_train)
        predict_test = grid.predict(X_test)
   
        self.model = grid
        self.print_metrics(grid, model_name,predict_train, predict_test, X_train, y_train, X_test, y_test)
        



    def trainSVR(self):
        regr = svm.SVR()
        model_name = type(regr).__name__
        print("############{}################".format(model_name))
        
        X_train, X_test, y_train, y_test = self.ts_train_test_split()
        parameters = { 'C': [0.1,1, 5, 10,100], 'epsilon': [0.01,  0.001,0.0001]}
        
        #parameters = { 'C': [0.01, 0.05, 0.1,1, 5,7, 10,20,100], 'epsilon': [0.005, 0.01, 0.02, 0.05,0.07,0.1,0.2,0.5,  0.001,0.002,0.003,]}
        #parameters = {'C': [1, 10], 'epsilon': [0.1, 1e-2, 1e-3]}
        r2_scorer = metrics.make_scorer(metrics.r2_score)
        
        grid = GridSearchCV(regr, param_grid=parameters, scoring=r2_scorer)
        grid.fit(X_train, y_train)
        
        predict_train = grid.predict(X_train)
        predict_test = grid.predict(X_test)

        
        self.model = grid
        self.print_metrics(grid, model_name,predict_train, predict_test, X_train, y_train, X_test, y_test)
    

    def trainKNN(self):
        n_neighbors = 5
        weights = 'uniform' # 'distance'
        #n_neighbors, weights=weights
        regr = neighbors.KNeighborsRegressor()
        model_name = type(regr).__name__
        print("############{}################".format(model_name))
        
        X_train, X_test, y_train, y_test = self.ts_train_test_split()
        parameters = {'n_neighbors': [2,3,4], 'weights': ['distance','uniform']}
        
        #parameters = {'n_neighbors': [2,3,4, 5,6, 7,8, 9 ,10], 'weights': ['distance','uniform']}
        
        r2_scorer = metrics.make_scorer(metrics.r2_score)

        grid = GridSearchCV(regr, param_grid=parameters, scoring=r2_scorer)
        grid.fit(X_train, y_train)
        
        predict_train = grid.predict(X_train)
        predict_test = grid.predict(X_test)

        self.model = grid
        
        self.print_metrics(grid, model_name,predict_train, predict_test, X_train, y_train, X_test, y_test)

    #predict 
    def make_predict(self,data):
        self.plot_fit(data)
        lags = self.scaler.transform(self.final_row_unscaled)
        lags = pd.DataFrame(lags)
        #print(self.model.predict(lags))
        predicted = self.model.predict(lags)[0]
        
        return predicted
        
    def plot_bollinger(self,df, ticker):
        plot_type = "BollingerBands"
        df['20d_ma'] = df[ticker].rolling(window=20, center=False).mean()
        df['20d_stddev'] = df[ticker].rolling( window=20, center=False).std()
        fig =plotter.gcf()
        
        plotter.title("{} - {}".format(plot_type ,ticker))
        plotter.plot(df['20d_ma'])
        # plotting bollinger bands
        df['boll_up'] = df['20d_ma'] + df['20d_stddev'] * 2
        plotter.plot()
        df['boll_dn'] = df['20d_ma'] - df['20d_stddev'] * 2

        plotter.plot(df['boll_up'])
        plotter.plot(df[ticker])
        plotter.plot(df['boll_dn'])
        
        # df1 = df[['Date','Close']]
        
        #plotter.plot(df)
        plotter.legend()
        #plotter.show()
        fig.savefig('{}_{}.png'.format(plot_type, re.sub(r'([^\s\w]|_)+', '', ticker)))
        plotter.clf()

    def do_testing(self):
        """
        To test actual and predicted accuracy
        """
        filename = '{}_{}.csv'.format("testing", re.sub(r'([^\s\w]|_)+', '', self.ticker))
        if self.fetch_remote:
            print("Reloading")
            data = quandl.Dataset(self.ticker).data(
                params={'start_date': self.start_date, 'end_date': self.query_date}).to_pandas()
            # Save as CSV to avoid re-query
            data.to_csv(filename)
            data = pd.read_csv(filename,parse_dates=[0], index_col='Date')
        else:
            print("Reading CSV...")
            data = pd.read_csv(filename,parse_dates=[0], index_col='Date')
        actual_price = data[self.metric][ self.query_date]
        print("Actual price at query date: {:.4f}".format(actual_price))
        end_date_price = data[self.metric][self.end_date]
        return actual_price, end_date_price 
        
    def print_metrics(self,grid, model_name,predict_train,predict_test, X_train, y_train, X_test, y_test):
        self.df_metrics.ix["{} R2 score - training set".format(model_name)]= "{:.4f}".format(r2_score(predict_train, y_train))
        self.df_metrics.ix["{} R2 score - test set".format(model_name)] =  "{:.4f}".format(r2_score(predict_test, y_test))
        self.df_metrics.ix["{} Mean squared error".format(model_name)] = "{:.4f}".format(mean_squared_error(y_test, grid.predict(X_test)))
        
        
        print(tabulate([["Best {} params".format(model_name), grid.best_params_],
                        ["R2 score - training set", "{:.4f}".format(r2_score(predict_train, y_train))],
                        ["R2 score - test set", "{:.4f}".format(r2_score(predict_test, y_test))],
                        ["Mean squared error", "{:.4f}".format(mean_squared_error(y_test, grid.predict(X_test)))]],headers = ["Metric", "Value"], tablefmt="fancy_grid"))
        
    def print_results(self, actual_price,predicted_price,end_date_price):
        
        model_name = type(self.model.best_estimator_).__name__
        self.df_metrics.ix["Actual price"] = actual_price 
        self.df_metrics.ix["Predicted price({})".format(model_name)] = predicted_price
        self.df_metrics.ix["Percentage difference({})".format(model_name)] = "{:.4f}%".format(abs((actual_price-predicted_price)/actual_price)*100.0)
        
        
        print(tabulate([["Actual price", actual_price],
                        ["Predicted price", predicted_price],
                        ["Percentage difference", "{:.4f}%".format(abs((actual_price-predicted_price)/actual_price)*100.0)]], 
                        headers = [self.ticker, "Results"], tablefmt="fancy_grid"))
        return self.df_metrics
        
    def plot_fit(self, df):
        """
        Plotting fit graphs
        """
        fig =plotter.gcf()
        fig.suptitle("{} - {}".format(type(self.model.best_estimator_).__name__,self.ticker))
        
        new_array = np.array(df.index.to_pydatetime(), dtype=np.datetime64)
        predicted = None
        
        for i, row in self.unscaled_data.iterrows():
            lags = self.scaler.transform(self.unscaled_data[i-1:i])
            lags = pd.DataFrame(lags)
            if predicted is None:
                predicted = self.model.predict(lags)[0]
            else:
                predicted = np.append(predicted, self.model.predict(lags)[0])
        plotter.scatter(new_array[(new_array.shape[0]-predicted.shape[0]):],self.data["Adj. Close"][(self.data.shape[0]-predicted.shape[0]):],color='red')
        #plotting the initial datapoints 
        
        plotter.plot( new_array[(new_array.shape[0]-predicted.shape[0]):],predicted,color='blue',linewidth=3) #plotting the line made by linear regression
        #plt.show()
        fig.savefig('{}_{}.png'.format(type(self.model.best_estimator_).__name__, re.sub(r'([^\s\w]|_)+', '_fit_', self.ticker)))
        plotter.clf()
