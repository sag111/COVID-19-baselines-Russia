import joblib
import pandas as pd
from itertools import islice
from sklearn.base import BaseEstimator

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import AutoARIMA

from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def sliding_window(seq_x, seq_y=None, win_size=2, win_pred=1):
    if seq_y is None:
        seq_y = seq_x.copy()
        
    if len(seq_x) < win_size+win_pred:
        raise BadException({
            'reason': 'Too few seq', 
            'actual': len(seq_x), 
            'required': win_size+win_pred
        })

    it = iter(seq_x[:-win_pred])
    result = list(islice(it, win_size))
    
    if len(result) == win_size:
        yield result, seq_y[win_size-1+win_pred]
    
    for n_elem, elem in enumerate(it):
        result = result[1:] + [elem,]
        pred_val = seq_y[n_elem+win_size+win_pred]
        
        yield result, pred_val


class BadException(Exception):
    pass


class CovidData_Split(object):
    def __init__(self, n_splits=2, forward_chaining=True):
        self.forward_chaining = forward_chaining
        self.n_splits = n_splits
        
        if self.n_splits<2:
            raise BadException({
                'reason': 'Too small split numbers (Must be at least 2)', 
                'actual': n_splits, 
                'required': '2 or more'
            })
    
    def split_by_sliding_window(self, df, y_name):
        if y_name not in df.columns:
            raise BadException({
                'reason': 'This "y_name" is not in DataFrame columns', 
                'actual': y_name, 
                'required': '"y_name" must be in Data Frame columns'
            })

        for n_fold in range(self.n_splits-1):        
            y = df.loc[:,y_name]
            fold_days = len(y)//self.n_splits
            train_inds, test_inds = [], []

            if self.forward_chaining == False:
                train_inds = y.iloc[n_fold*fold_days:(n_fold+1)*fold_days].index.tolist()
            else:
                train_inds = y.iloc[0:(n_fold+1)*fold_days].index.tolist()
            
            if n_fold != self.n_splits-2:
                test_inds = y.iloc[(n_fold+1)*fold_days:(n_fold+2)*fold_days].index.tolist()
            else:
                test_inds = y.iloc[(n_fold+1)*fold_days:].index.tolist()

            yield train_inds, test_inds


class StatsModelsRegression(object):
    def __init__(self, model_name='ExpSmoothing'):
        self.model_name = model_name
        if self.model_name == 'ExpSmoothing':
            self.model = ExponentialSmoothing
        elif self.model_name == 'AutoARIMA':
            self.model = AutoARIMA
        else:
            raise BadException({
                'reason': 'This model_name is not supported',
                'actual': self.model_name, 
                'required': ['AutoARIMA', 'ExpSmoothing']
            })
    
    def fit(self, X):
        if self.model_name == 'AutoARIMA':
            self.fit_model = self.model(start_p=0, d=None, start_q=0, max_p=5, max_d=2, max_q=5, 
                                        start_P=0, D=None, start_Q=0, max_P=3, max_D=2, max_Q=3,
                                        max_order=5, m=1, seasonal=True, stationary=False,).fit(y=X)
        
        elif self.model_name == 'ExpSmoothing':
            self.fit_model = self.model(endog=X, trend='add').fit()
        
        return self.fit_model
    
    def predict(self, pred_len):
        if self.model_name == 'AutoARIMA':                        
            pred_seq = self.fit_model.predict(n_periods=pred_len)
        else:
            pred_seq = self.fit_model.forecast(pred_len)
        
        return pred_seq


class ModelsRegression(BaseEstimator):
    def __init__(self, configs):
        self.configs = configs
        self.model_name = configs['model_name']
        
        regression_sklearn_models_zoo = {
            'dummy_model': None,
            'LinearSVR': LinearSVR(max_iter=5000),
            'LR': LinearRegression(normalize=True),
            'Ridge': Ridge(normalize=True),
            'Lasso': Lasso(max_iter=5000, normalize=True),
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'GBR': GradientBoostingRegressor(n_estimators=100),
        }
        
        if self.model_name not in ['dummy_model', 
                                   'LinearSVR', 'LR', 'Ridge', 'Lasso', 'RandomForest', 
                                   'AutoARIMA', 'ExpSmoothing']:
            raise BadException({
                'reason': 'This model_name is not supported', 
                'actual': self.model_name, 
                'required': 'dummy_model, LinearSVR, LR, Ridge, Lasso, RandomForest, AutoARIMA, ExpSmoothing',
            })
            
        elif self.model_name in regression_sklearn_models_zoo.keys():
            self.model = regression_sklearn_models_zoo[self.model_name]
        
        else:
            self.model = StatsModelsRegression(model_name=self.model_name)
        
        
    def fit_predict(self, df_region, train_index, test_index):
        true_y = df_region.loc[test_index, self.configs['y_name']].tolist()
        
        # naive approach
        if self.model_name == 'dummy_model':
            train_y = df_region.loc[train_index, self.configs['y_name']].tolist()
            pred_y = train_y[-self.configs['prediction_depth']:]+true_y[:-self.configs['prediction_depth']]

        # stat models
        elif self.model_name in ['ExpSmoothing', 'AutoARIMA']:
            pred_y = []
            for ind_y, y_i in enumerate(true_y):
                if ind_y < self.configs['prediction_depth']-1:
                    train_y = df_region.loc[train_index[:-self.configs['prediction_depth']+ind_y], self.configs['y_name']].tolist()
                else:
                    train_y = df_region.loc[train_index+test_index[:ind_y+1-self.configs['prediction_depth']], self.configs['y_name']].tolist()
                    
                self.model.fit(train_y)
                pred_val = self.model.predict(self.configs['prediction_depth'])[-1]
                pred_y.append(pred_val)

        # ML models
        else:
            # train ML model
            train_x = df_region.loc[train_index, self.configs['x_name']].tolist()
            train_y = df_region.loc[train_index, self.configs['y_name']].tolist()
            
            X, y = [], []
            for row, pred_val in sliding_window(train_x, train_y, 
                                                win_size=self.configs['history_size'], 
                                                win_pred=self.configs['prediction_depth']):
                X.append(row)
                y.append(pred_val)
            model = self.model.fit(X, y)
        
            # test ML model
            test_x = df_region.loc[test_index, self.configs['x_name']].tolist()
            test_y = df_region.loc[test_index, self.configs['y_name']].tolist()
            
            # add some samples (history_size+prediction_depth-1) from train data 
            # for start predict from first test sample
            test_x = train_x[-(self.configs['history_size']+self.configs['prediction_depth'])+1:]+test_x
            test_y = train_y[-(self.configs['history_size']+self.configs['prediction_depth'])+1:]+test_y
            
            X, true_y = [], []
            for row, pred_val in sliding_window(test_x, test_y,
                                                win_size=self.configs['history_size'],
                                                win_pred=self.configs['prediction_depth']):
                X.append(row)
                true_y.append(pred_val)
            pred_y = model.predict(X)
                
        return true_y, pred_y
    
    def save_model(self, model_path):
        return joblib.dump(self, model_path)
    
    @staticmethod
    def load_model(model_path):
        return joblib.load(model_path)