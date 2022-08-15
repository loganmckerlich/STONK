
# grid search sarima hyperparameters
import time
from math import sqrt
from multiprocessing import cpu_count
from warnings import catch_warnings, filterwarnings
import yfinance as yf
import numpy as np
 
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
 
# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
 
# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))
 
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]
 
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error
 
# score a model, return None on failure
def score_model(data, n_test, cfg, step, length, debug=False):
    step=step+1
    start = time.perf_counter()
    result = None
    elapse = 0
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        elapse = (time.perf_counter() - start)
        print(f'{step}/{length}')
        print(' > Model[%s] %.3f time: %.3fs' % (key, result, elapse))
    return (key, result, elapse)
 
# grid search configs
def grid_search(data, cfg_list, n_test, length, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')  
        tasks = (delayed(score_model)(data, n_test, cfg, cfg_list.index(cfg), length) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg, cfg_list.index(cfg), length) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores
 
# create a set of sarima configs to try
def sarima_configs(seasonal,p_,d_,q_,t_,P_,D_,Q_):
    models = list()
    # define config lists
    p_params = p_
    d_params = d_
    q_params = q_
    t_params = t_
    P_params = P_
    D_params = D_
    Q_params = Q_
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    length = len(models)
    print(length)
    return models, length
 

def g_test(data, n_test,seasonal,p_,d_,q_,t_,P_,D_,Q_):
    print(f'data is {len(data)} rows')
    print('starting')
    cfg_list,length = sarima_configs(seasonal,p_,d_,q_,t_,P_,D_,Q_)
    scores = grid_search(data, cfg_list, n_test, length)
    for cfg, error, elapse in scores[:3]:
        print(cfg, error, elapse)
    return scores[:3][cfg]


if __name__ == '__main__':
   
    # define dataset
    data = np.array(yf.Ticker('PCAR').history(period='24mo')[['Close']])
    # data split
    n_test = 14
    seasonal=[5,7,30,365]
    p_=[5,7,30]
    d_=[0,1,2]
    q_=[0,1]
    t_ = ['n','c','t','ct']
    P_=[0,5]
    D_=[0,1]
    Q_=[0,1]
    top = g_test(data, n_test,seasonal,p_,d_,q_,t_,P_,D_,Q_)

       