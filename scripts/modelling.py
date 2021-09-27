import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor


class Model:
    def __init__(self, wdir: str, mode: str) -> None:
        self.wdir = wdir
        self.mode = mode
        self.maindf = None
        self.x = None
        self.y = None
        self.index_sets = None
        self.trained_model = None

    def set_data(self, data_fname: str):
        with open(os.path.join(self.wdir, 'temp', data_fname), 'rb') as f:
            self.maindf = pickle.load(f)
        if self.mode == 'train':
            self.x = self.maindf.drop('rating', axis=1)
            self.y = self.maindf['rating']
        elif self.mode == 'predict':
            self.x = self.maindf
            self.y = None

    def load_model(self, model_fname: str):
        with open(os.path.join(self.wdir, 'assets', model_fname), 'rb') as f:
            self.trained_model = pickle.load(f)
        print("--- Model loaded ---")

    def save_model(self, model_fname: str):
        with open(os.path.join(self.wdir, 'assets', model_fname), 'wb') as f:
            pickle.dump(self.trained_model, f)
        with open(os.path.join(self.wdir, 'assets', "feature_names.pkl"), 'wb') as f:
            pickle.dump(self.x.columns, f)
        print("--- Model saved ---")
        print("--- Names saved ---")

    def set_folds(self):
        kf = KFold(n_splits=5)
        self.index_sets = kf.split(self.x, self.y)

    def set_model(self, model_code: str):
        if model_code == 'linear':
            curr_model = LinearRegression()
        elif model_code == 'rf':
            curr_model = RandomForestRegressor(100)
        elif model_code == 'lightgbm':
            curr_model = LGBMRegressor(n_estimators=500)
        else:
            print(f"No valid entry for the model. Using Random Forest")
            curr_model = RandomForestRegressor(100)
        return curr_model

    def train(self, model_code: str) -> pd.DataFrame:
        if self.x is None or self.y is None:
            print(f"!!! No data is loaded !!!")
            return None
        curr_model = self.set_model(model_code)
        fold, metrics = 1, []
        print("Starting training")
        for (tr_ind, val_ind) in self.index_sets:
            x_train = self.x.drop('record_id', axis=1).iloc[tr_ind, :]
            y_train = self.y.iloc[tr_ind]
            x_val = self.x.drop('record_id', axis=1).iloc[val_ind, :]
            y_val = self.y.iloc[val_ind]
            curr_model.fit(x_train.values, y_train.values)
            y_pred = curr_model.predict(x_val)
            r2 = round(r2_score(y_val, y_pred), 2)
            rmse = round(np.sqrt(mean_squared_error(y_val, y_pred)), 2)
            mae = round(mean_absolute_error(y_val, y_pred), 2)
            print(f"Fold : {fold} | R2 : {r2} | RMSE : {rmse} | MAE : {mae}")
            metrics.append([r2, rmse, mae])
            fold += 1
        self.trained_model = curr_model
        print("Training finished")
        return pd.DataFrame(metrics, columns=['r2-score', 'rmse', 'mae'])

    def get_predictions(self) -> np.array:
        if self.x is None:
            print(f"!!! No data is loaded !!!")
            return None
        if self.trained_model is None:
            print(f"!!! No model is loaded !!!")
            return None
        with open(os.path.join(self.wdir, 'assets', "feature_names.pkl"), 'rb') as f:
            col_names = pickle.load(f)
        cols_to_eliminate = [x for x in self.x.columns if x not in col_names]
        self.x.drop(cols_to_eliminate, axis=1, inplace=True)
        cols_to_fill = [x for x in col_names if x not in self.x.columns]
        zero_data = pd.DataFrame(
            np.zeros((self.x.shape[0], len(cols_to_fill))), columns=cols_to_fill)
        self.x = pd.concat([self.x, zero_data], axis=1)
        self.x = self.x[col_names]
        y_pred = self.trained_model.predict(self.x.drop('record_id', axis=1))
        return y_pred
