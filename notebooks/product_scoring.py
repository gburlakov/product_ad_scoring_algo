#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def rank_products(all_predictions_df, metrics_df, all_ranked_products_filepath):
    """
    This function ranks the product IDs based on predicted values of the target variable
    """

    df = all_predictions_df.reset_index().rename(columns={'index':'product_id'})
    
    ranked_list = []

    combos = df[['category','target','ad_platform']].drop_duplicates()

    for _, combo in combos.iterrows():
        cat, tgt, plat = combo['category'], combo['target'], combo['ad_platform']
        
        mask = (
            (metrics_df['category']   == cat) &
            (metrics_df['target']     == tgt) &
            (metrics_df['ad_platform']== plat) &
            (metrics_df['rank']       == 1)
        )
        best_models = metrics_df.loc[mask, 'model'].unique()
        if len(best_models) == 0:
            # no entry — skip
            continue
        best_model = best_models[0]

        df_sub = df[
            (df['category']    == cat) &
            (df['target']      == tgt) &
            (df['ad_platform'] == plat) &
            (df['model']       == best_model)
        ].copy()
        
        df_sub = df_sub.sort_values('pred_value', ascending=False)
        df_sub['rank'] = np.arange(1, len(df_sub) + 1)
        
        ranked_list.append(
            df_sub[['product_id','category','target','ad_platform','model','pred_value','actual','rank']]
        )
    
    all_ranked_products_df = pd.concat(ranked_list, ignore_index=True)
    
    all_ranked_products_df.to_csv(all_ranked_products_filepath, index=False)
    
    return all_ranked_products_df


# In[ ]:


# def rank_products(all_predictions_df, metrics_df):
#     """
#     This function ranks the product IDs based on predicted values of the target variable
#     """
#     ranked_list = []
#     combos = metrics_df[metrics_df['rank']==1][['target','category','ad_platform','model']]
#     for _, row in combos.iterrows():
#         df_sub = all_predictions_df[
#             (all_predictions_df['target']==row['target']) &
#             (all_predictions_df['category']==row['category']) &
#             (all_predictions_df['ad_platform']==row['ad_platform']) &
#             (all_predictions_df['model']==row['model'])
#         ].copy()

#         df_sub = df_sub.sort_values('pred_value', ascending=False)
#         df_sub['rank'] = np.arange(1, len(df_sub)+1)
#         ranked_list.append(df_sub[['product_id','category','target','ad_platform','model','pred_value','actual','rank']])
#     all_ranked_products_df = pd.concat(ranked_list, ignore_index=True)
    
#     return all_ranked_products_df


# In[ ]:


# def score_products(df, predictions_df, validation_indices):
#     """
#     Scores products based on predicted ROI and ranks them, adding an ad_platform label and individual model predictions.

#     Args:
#         df (pd.DataFrame): dataframe containing product data (with 'price' column)
#         predictions_df (pd.DataFrame): dataframe containing individual model predictions and the ensemble prediction
#         validation_indices (pd.Index): Index of the validation set
#         ad_platform (str): ad platform, 'meta' or 'google'
#     """
#     predictions_df = pd.DataFrame({f'{target}': f'{target}'}, index=validation_indices)
    
#     return predictions_df    


# In[ ]:


# def score_products(df, predictions_df, validation_indices, ad_platform):
#     """
#     Scores products based on predicted ROI and ranks them, adding an ad_platform label and individual model predictions.

#     Args:
#         df (pd.DataFrame): dataframe containing product data (with 'price' column)
#         predictions_df (pd.DataFrame): dataframe containing individual model predictions and the ensemble prediction
#         validation_indices (pd.Index): Index of the validation set
#         ad_platform (str): ad platform, 'meta' or 'google'
#     """

#     scored_df = df.join(predictions_df, how="inner")

#     for col in predictions_df.columns:
#         scored_df[col + '_score'] = scored_df[col] * scored_df['price']

#     for col in predictions_df.columns:
#         scored_df[col + '_rank'] = scored_df[col + '_score'].rank(ascending = False)

#     scored_df['ad_platform'] = ad_platform

#     return scored_df.sort_values('ensemble_prediction_rank')


# In[ ]:


# def score_products(df, roi_predictions, validation_indices, ad_platform):
#     """
#     Scores products based on predicted ROI and ranks them, adding an ad_platform label.

#     Args:
#         df (pd.DataFrame): dataFrame containing product data (with 'price' column).
#         roi_predictions (np.ndarray): Array of predicted ROI values for the validation set.
#         validation_indices (pd.Index): Index of the validation set.
#         ad_platform (str): The name of the ad platform ('meta' or 'google').

#     Returns:
#         pd.DataFrame: DataFrame with predicted ROI, score, rank, and ad_platform label.
#     """

#     predictions = pd.DataFrame({'predicted_roi': roi_predictions}, index=validation_indices)

#     scored_df = df.join(predictions, how="inner")  # Inner join to only keep items with a valid ROI

#     scored_df['score'] = scored_df['predicted_roi'] * scored_df['price']  # Example: ROI * Price

#     scored_df['rank'] = scored_df['score'].rank(ascending=False)

#     scored_df['ad_platform'] = ad_platform

#     return scored_df.sort_values('rank')

