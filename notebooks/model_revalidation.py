#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


def compute_metrics_by_category(all_predictions_df, all_validation_metrics_filepath):
    """
    This function re-assesses models based on the validation dataset
    """
    results = []

    combos = all_predictions_df[['target','category','ad_platform','model']].drop_duplicates()
    for _, row in combos.iterrows():
        df_sub = all_predictions_df[
            (all_predictions_df['target']==row['target']) &
            (all_predictions_df['category']==row['category']) &
            (all_predictions_df['ad_platform']==row['ad_platform']) &
            (all_predictions_df['model']==row['model'])
        ]
        y_true = df_sub['actual']
        y_pred = df_sub['pred_value']
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        results.append({**row.to_dict(), 'RMSE': rmse, 'MAE': mae})

    all_combo = all_predictions_df[['target','ad_platform','model']].drop_duplicates()
    for _, row in all_combo.iterrows():
        df_sub = all_predictions_df[
            (all_predictions_df['target']==row['target']) &
            (all_predictions_df['ad_platform']==row['ad_platform']) &
            (all_predictions_df['model']==row['model'])
        ]
        y_true = df_sub['actual']
        y_pred = df_sub['pred_value']
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        results.append({
            'target': row['target'],
            'category': 'all_categories',
            'ad_platform': row['ad_platform'],
            'model': row['model'],
            'RMSE': rmse,
            'MAE': mae
        })
    metrics_df = pd.DataFrame(results)

    metrics_df['rank'] = metrics_df.groupby(
        ['target','category','ad_platform']
    )['RMSE'].rank(method='first', ascending=True)
    
    return metrics_df

