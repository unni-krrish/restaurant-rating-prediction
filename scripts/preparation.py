import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# When fitting OHE again on unseen data, extra elements may come
# This will result in null values - take care of this...!!!

class DataPrepare:
    def __init__(self, wdir, filename) -> None:
        self.wdir = wdir
        with open(os.path.join(self.wdir, 'temp', filename), 'rb') as f:
            self.df = pickle.load(f)

    def ohe_encoder(self, col):
        ohe = OneHotEncoder(sparse=False)
        encoded = ohe.fit_transform(self.df[col].values.reshape(-1, 1))
        ret_df = pd.DataFrame(
            data=encoded,
            columns=[f"{col}_{i}" for i in list(ohe.categories_[0])])
        return ret_df

    def get_one_hot_features(self):
        cols = ['listing_type', 'locality', 'rest_type']
        for col in cols:
            encoded = self.ohe_encoder(col)
            self.df = pd.concat([self.df, encoded], axis=1)
            self.df.drop(col, axis=1, inplace=True)

    def get_binary_features(self):
        cols = ['online_order', 'book_table']
        for col in cols:
            self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})

    def normalize_features(self):
        cols = ['cost_for_two', 'rating']
        for col in cols:
            scaler = StandardScaler()
            self.df[col] = scaler.fit_transform(self.df[col])

    def treat_col_cuisines(self):
        all_cuisines = []
        for item in self.df['cuisines'].unique():
            try:
                sub_items = item.split(',')
                for sub_item in sub_items:
                    if sub_item.strip() not in all_cuisines:
                        all_cuisines.append(sub_item.strip())
            except AttributeError:
                print("Dealing with nan value")

        arr = np.zeros((self.df.shape[0], len(all_cuisines)))
        for i in range(arr.shape[0]):
            try:
                for item in self.df['cuisines'][i].split(','):
                    elem = item.strip()
                    ind = all_cuisines.index(elem)
                    arr[i, ind] = 1
            except AttributeError:
                print("Dealing with nan value")
        newdf = pd.DataFrame(arr, columns=all_cuisines)
        self.df = pd.concat([self.df, newdf], axis=1)
        self.df.drop('cuisines', axis=1, inplace=True)

    def save_data(self, fname):
        self.df.dropna(inplace=True)
        self.df.reset_index(inplace=True)
        with open(os.path.join(self.wdir, 'temp', fname), 'wb') as f:
            pickle.dump(self.df, f)
        print("--- Saved data to local ---")
