import os
from cleaning import etl
from preparation import prepare_data
from modelling import model


curr_dir = os.path.abspath('')

etl = etl(curr_dir, 'zomato.csv')
etl.treat_columns()
etl.treat_rows()
etl.treat_dtypes()
etl.save_data("t1_cleaned.pkl")

prep = prepare_data(curr_dir, 't1_cleaned.pkl')
prep.get_binary_features()
prep.get_one_hot_features()
prep.treat_col_cuisines()
prep.save_data("t1_prepared.pkl")

model = model(curr_dir, 'train')
model.set_data("t1_prepared.pkl")
model.set_folds()
summary = model.train('lightgbm')
model.save_model("t1_model_lightgbm.pkl")
print(summary)
