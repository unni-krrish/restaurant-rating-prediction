import os
from cleaning import DataCleaner
from preparation import DataPrepare
from modelling import Model
import numpy as np
import pandas as pd
curr_dir = os.path.abspath('')


# Train using zomato.csv in data folder
batch_name = 'train_1'
etl_train = DataCleaner(curr_dir, 'zomato.csv')
etl_train.treat_columns()
etl_train.treat_rows()
etl_train.treat_dtypes()
etl_train.treat_rating()
etl_train.save_data(batch_name + "_cleaned.pkl")

prep = DataPrepare(curr_dir, batch_name + '_cleaned.pkl')
prep.get_binary_features()
prep.get_one_hot_features()
prep.treat_col_cuisines()
prep.save_data(batch_name + "_prepared.pkl")

train_model = Model(curr_dir, 'train')
train_model.set_data(batch_name + "_prepared.pkl")
train_model.set_folds()
summary = train_model.train('lightgbm')
train_model.save_model(batch_name + "_model_lightgbm.pkl")
print(summary)

# Prediction on a subset of zomato.csv
batch_name = 'test_1'
etl_pred = DataCleaner(curr_dir, 'test_set_1.csv')
etl_pred.treat_columns()
etl_pred.treat_rows()
etl_pred.treat_dtypes()
etl_pred.save_data(batch_name + "_cleaned.pkl")

prep = DataPrepare(curr_dir, batch_name + '_cleaned.pkl')
prep.get_binary_features()
prep.get_one_hot_features()
prep.treat_col_cuisines()
prep.save_data(batch_name + "_prepared.pkl")

pred_model = Model(curr_dir, 'predict')
pred_model.set_data(batch_name + '_prepared.pkl')
pred_model.load_model('train_1_model_lightgbm.pkl')
y_pred = pred_model.get_predictions()

# Final Results in a dataframe
final_res = np.concatenate(
    [pred_model.maindf['record_id'].values.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
final_res_df = pd.DataFrame(
    final_res, columns=['record_id', 'predicted_rating'])
final_res_df['record_id'] = final_res_df['record_id'].astype(int)
