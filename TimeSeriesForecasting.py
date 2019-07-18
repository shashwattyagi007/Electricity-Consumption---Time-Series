import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings,itertools,matplotlib
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from pylab import rcParams

warnings.filterwarnings("ignore")

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# label encoding of column var2
le = preprocessing.LabelEncoder()
le.fit(train['var2'])
train['var2'] = le.transform(train['var2']) 

train['year'] = train['datetime'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").year)
train['month'] = train['datetime'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").month)
train['day'] = train['datetime'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").day)
train['hour'] = train['datetime'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").hour)
train['weekday'] = train['datetime'].apply(lambda x: datetime.strptime(x,"%Y-%m-%d %H:%M:%S").weekday())

def get_optimum_param(data):
    """
    Returns optimum parameter for minimu AIC
    input : time series data
    output: param, param_seasonal
    """
    min_aic = 1000000000
    min_param = None
    min_param_seasonal = None
    
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                if results.aic < min_aic:
                    min_aic = results.aic
                    min_param = param
                    min_param_seasonal = param_seasonal
            except:
                continue
    print('\n\nMinimum AIC {} Minimum param {} Minimum param seasonal {}\n\n'.format(min_aic
                                                                                     ,min_param
                                                                                     ,min_param_seasonal))
    return min_param,min_param_seasonal

def train_model(data,param , param_seasonal):
    """
    Returns timeseries model for forecasting
    input: data,param , param_seasonal
    output: model
    """
    mod = sm.tsa.statespace.SARIMAX(data,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    return results

def model_evaluation(data,model,year, month, date):
    """
    Returns model validation results and plot 
    input : data = timeseries data
            model = time series model
            year = 2013
            month= 07 
            date = 24
    """
    startdate = str(year) + '-' + str(month) + '-' + str(date)
    pred = model.get_prediction(start=pd.to_datetime(startdate), dynamic=False)
    pred_ci = pred.conf_int()
    ax = data[str(year):].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('y')
    plt.legend()
    plt.show()
    
    return pred_ci

def append_next_month_data(data,pred_ci,year,month):
    try:
        mean = (pred_ci['upper electricity_consumption'] + pred_ci['lower electricity_consumption'])/2   
    except:
        mean = (pred_ci['upper y'] + pred_ci['lower y'])/2
    
    new_data = pd.concat([data,mean,train[(train['year'] == year) & (train['month'] == month) ]['electricity_consumption']])
    new_data.plot(figsize=(15, 6))
    plt.show()
    return new_data

def get_forecast(model,data,steps = 192):
    pred_uc = model.get_forecast(steps=steps)
    pred_ci = pred_uc.conf_int()
    ax = data.plot(label='observed', figsize=(14, 7))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('electricity_consumption')
    plt.legend()
    plt.show()
    return pred_ci

def show_components(new_data):
    """
    Returns decomposition plot for a timeseries data
    input: timeseries data
    output: observed, trend, seasality and residual plot
    """
    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(new_data, model='additive')
    fig = decomposition.plot()
    plt.show()
    
def number_of_steps(year, month):
    """
    Returns number of steps to forecast for given year and month
    input : year, month
    output: steps
    """
    if (month == 4 or month == 6 or month == 9 or month == 11):
        steps = 168
        return steps
    
    elif (month == 2):
        if (year % 4) == 0:
            if (year % 100) == 0:
                if (year % 400) == 0:
                    steps = 144
                    return steps
                else:
                    steps = 120
                    return steps
            else:
                steps = 144
                return steps
        else:
            steps = 120
            return steps
        
    else:
        steps = 192
        return steps


train.Timestamp = pd.to_datetime(train.datetime,format="%Y-%m-%d %H:%M:%S") 
train.index = train.Timestamp
temp1 = train.resample('H').mean()
temp1 = train[(train['year'] == 2013) & (train['month'] == 7) ]['electricity_consumption']

data = temp1
date = 24

result = pd.DataFrame()
model_res = pd.DataFrame()

for year in train['year'].unique():
    for month in range(1,13):
        if ((year == 2013) & (month < 7)) or ((year == 2017) & (month > 6)):
            pass
        else:
            print(year,month)
            param,param_seasonal = get_optimum_param(data)
            print('training model from ',year,month)
            model = train_model(data,param,param_seasonal)

            steps = number_of_steps(year, month)
            
            pred_ci = get_forecast(model,data,steps) # find formula to calculate number of steps
            next_month =   (month + 1)%12
            next_year = year
            if next_month == 0:
                next_month = 12
            if next_month == 1:
                next_year = year + 1
            print('adding data of ',next_year,next_month)
            new_data = append_next_month_data(data,pred_ci,next_year,next_month)
            data = new_data
            show_components(new_data)
            result = result.append(pred_ci, ignore_index=True)
        print('length of result: ',len(result))
    result.to_csv('result.csv')






