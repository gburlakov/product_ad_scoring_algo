#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt


# In[2]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ttest_ind


# In[3]:


from config import input_data_folderpath, input_data_xlsx_filepath, input_data_csv_filepath, date_column, date_prefix_list, trained_models_folderpath, \
date_column, ad_platform_list, target_dict, train_size, feature_list, model_dict, save_val_end_points, n_trials_optuna, all_predictions_csv_filepath, \
all_validation_metrics_filepath, all_ranked_products_filepath, all_actual_products_selected_filepath, all_pred_products_selected_filepath, all_ab_test_results_filepath
from data_load import load_data
from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import set_train_test_val_size, train_model
from metrics_prediction import predict_metric
from model_revalidation import compute_metrics_by_category
from product_scoring import rank_products
from AB_testing import ab_test_selection


# In[4]:


input_data_folderpath


# In[5]:


# load_data(input_data_xlsx_filepath, input_data_csv_filepath)
input_df = pd.read_csv(input_data_csv_filepath)
input_df.set_index('product_id', inplace=True)
print('Preprocessing data...')
preprocessed_df = preprocess_data(input_df, date_column)
print('Feature engineering...')
featured_df = engineer_features(preprocessed_df, date_column, date_prefix_list, ad_platform_list, input_data_folderpath)
val_first_day, val_last_day, val_last_day_str = save_val_end_points(featured_df.copy())

dummy_columns = [col for col in featured_df.columns if any(col.startswith(prefix) for prefix in date_prefix_list)]
features = feature_list + dummy_columns

predictions_df_list = []
for ad_platform in ad_platform_list:
    for target in target_dict[ad_platform]:            
        print('Model training...')
        orig_X_val, orig_y_val = set_train_test_val_size(featured_df, val_first_day, val_last_day, train_size, ad_platform, target, features, 
                                                         date_column, date_prefix_list, 
                                                         dummy_columns, encoding=False)        
        X_train, X_test, X_val, y_train, y_test, y_val, encoders, scalers = set_train_test_val_size(featured_df, val_first_day, 
                                                                                                    val_last_day, train_size, 
                                                                                                    ad_platform, target, features, 
                                                                                                    date_column, date_prefix_list, 
                                                                                                    dummy_columns, encoding=True)
        train_model(featured_df, ad_platform, target, features, val_last_day_str, 
                    trained_models_folderpath, X_train, X_test, y_train, y_test, X_val, y_val, encoders, scalers, model_dict, 
                    n_trials_optuna)
        print('Metrics prediction...')
        prediction_df = predict_metric(trained_models_folderpath, val_last_day_str, X_val, y_val, orig_X_val, target, ad_platform, model_dict)
        predictions_df_list.append(prediction_df)

all_predictions_df = pd.concat(predictions_df_list)
all_predictions_df.to_csv(all_predictions_csv_filepath, index=True)

print('Model Validation...')
metrics_df = compute_metrics_by_category(all_predictions_df, all_validation_metrics_filepath)

print('Product Scoring...')
all_ranked_products_df = rank_products(all_predictions_df, metrics_df, all_ranked_products_filepath)

print('AB testing...')
all_actual_products_selected_df, all_pred_products_selected_df, all_ab_test_results_df = ab_test_selection(all_ranked_products_df, all_predictions_df,\
                                                                                                           all_actual_products_selected_filepath, \
                                                                                                           all_pred_products_selected_filepath, \
                                                                                                           all_ab_test_results_filepath)


# In[ ]:


# all_actual_products_selected_df = all_actual_products_selected_df.drop_duplicates()
# all_pred_products_selected_df = all_pred_products_selected_df.drop_duplicates()
# all_actual_products_selected_df.to_csv(all_actual_products_selected_filepath, index=False)
# all_pred_products_selected_df.to_csv(all_pred_products_selected_filepath, index=False)


# In[ ]:


# metrics_df = compute_metrics_by_category(all_predictions_df, all_validation_metrics_filepath)


# In[ ]:


# metrics_df


# In[ ]:


# all_ranked_products_df = rank_products(all_predictions_df, metrics_df, all_ranked_products_filepath)


# In[ ]:


# all_actual_products_selected_df, all_pred_products_selected_df, all_ab_test_results_df = ab_test_selection(all_ranked_products_df, all_predictions_df,\
#                                                                                                            all_actual_products_selected_filepath, \
#                                                                                                            all_pred_products_selected_filepath, \
#                                                                                                            all_ab_test_results_filepath)

