import os
import pickle
import numpy as np
import pandas as pd


class DataCleaner:
    def __init__(self, wdir: str, filename: str) -> None:
        self.wdir = wdir
        f_path = os.path.join(wdir, 'data', filename)
        self.df = pd.read_csv(f_path)
        self.check_columns()
        self.create_ids()

    def check_columns(self):
        with open(os.path.join(self.wdir, 'assets', 'original_cols.pkl'), 'rb') as f:
            original_cols = pickle.load(f)
        cols_absent = [
            col for col in original_cols if col not in self.df.columns]
        if len(cols_absent) > 0:
            print(
                f"Following columns are not found in the input data : \n {cols_absent}")
            self.df = None

    def create_ids(self):
        self.df['record_id'] = np.arange(0, self.df.shape[0])

    def get_summary(self) -> pd.DataFrame:
        lst = []
        for col in self.df.columns:
            lst.append([col, self.df[col].dtype,
                        self.df[col].nunique(),
                        self.df[col].isnull().sum(),
                        self.df[col].unique() if self.df[col].nunique() <= 5 else 'large'])

        summary = pd.DataFrame(data=lst,
                               columns=['col', 'dtype', 'n_unique', 'n_nulls', 'uniques'])
        return summary.sort_values(['dtype', 'n_unique'])

    def treat_columns(self):
        # Rename columns for readability
        rename_dct = {'listed_in(city)': 'locality',
                      'listed_in(type)': 'listing_type',
                      'approx_cost(for two people)': 'cost_for_two'}
        self.df.rename(rename_dct, axis=1, inplace=True)

        # Drop unnecessary columns
        cols_to_drop = ['name', 'menu_item', 'address', 'location',
                        'phone', 'reviews_list', 'url', 'dish_liked']
        self.df.drop(cols_to_drop, axis=1, inplace=True)
        print(f"--- Renamed, dropped columns : {self.df.shape}")

    def convert_rating(self, x):
        try:
            return float(x.split('/')[0])
        except ValueError:
            return np.nan

    def convert_cost(self, x):
        try:
            return int(''.join(x.split(',')))
        except ValueError:
            return np.nan

    def treat_rows(self):
        # Drop rows based on null values of important and least useful columns
        cols = ['rest_type', 'cuisines', 'cost_for_two']
        for col in cols:
            if self.df[col].isnull().sum() > 0:
                self.df.dropna(subset=[col], inplace=True)
        print(f"--- Dropped appropriate rows : {self.df.shape}")

    def treat_dtypes(self):
        # Cost for two usually contains commas and other characters
        if pd.api.types.is_object_dtype(self.df['cost_for_two']):
            self.df['cost_for_two'] = self.df['cost_for_two'].apply(
                self.convert_cost)

        # Drop null values after treating the above two columns
        for col in ['cost_for_two', 'cuisines']:
            self.df.dropna(subset=[col], inplace=True)
        self.df.reset_index(drop=True)
        print(f"Done treating dtypes : {self.df.shape}")

    def treat_rating(self):
        self.df.rename({'rate': 'rating'}, axis=1, inplace=True)
        if self.df['rating'].isnull().sum() > 0:
            self.df.dropna(subset=['rating'], inplace=True)

        # Drop rows where the rating column is improper
        slicer = self.df['rating'].apply(
            lambda x: True if '/' in x else False)
        self.df = self.df[slicer].reset_index(drop=True)

        # Rating contains / and other characters
        if pd.api.types.is_object_dtype(self.df['rating']):
            self.df['rating'] = self.df['rating'].apply(self.convert_rating)

        # Drop na of rating
        self.df.dropna(subset=['rating'], inplace=True)

    def save_data(self, fname):
        with open(os.path.join(self.wdir, 'temp', fname), 'wb') as f:
            pickle.dump(self.df, f)
        print(f'before saving : {self.df.shape}')
        print("--- Saved data to local ---")
