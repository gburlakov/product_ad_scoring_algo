#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from config import date_column, date_prefix_list, ad_platform_list


# In[ ]:


def encode_date(df, date_column, date_prefix_list):
    """
    df (pd.DataFrame): dataframe with date column to be encoded
    date_column (str): date column name
    prefix_list (str): list of prefixes of date dummies
    """
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['week'] = df[date_column].dt.isocalendar().week.astype(int)
    df['weekday'] = df[date_column].dt.weekday
    
    return pd.get_dummies(df, columns=['year', 'month', 'week', 'weekday'], prefix=date_prefix_list)


# In[3]:


def compute_discount(df):
    """
    This function derives the discount rate column from the price and sale price columns
    
    Args:
        df (pd.DataFrame): input dataframe with price columns
    """
    df['discount_rate'] = (df['price'] - df['sale_price']) / df['price']
    df['discount_rate'] = df['discount_rate'].fillna(0)
    
    return df


# In[4]:


def compute_target_metrics(df, ad_platform):
    """
    This function derives the target metrics from online perfomance columns by ad platform
    
    Args:
        df (pd.DataFrame): input dataframe with online perfomance columns
        ad_platform (str): ad platform, meta and google
    """
    df[f'{ad_platform}_impressions_per_spend'] = (df[f'{ad_platform}_impressions'] / (df[f'{ad_platform}_spend'])).fillna(0)
    
    df[f'{ad_platform}_clickthrough'] = (df[f'{ad_platform}_clicks'] / df[f'{ad_platform}_impressions']).fillna(0)
    df[f'{ad_platform}_clickthrough_per_spend'] = (df[f'{ad_platform}_clickthrough'] / (df[f'{ad_platform}_spend'])).fillna(0)
    
    df[f'{ad_platform}_conversion'] = (df[f'{ad_platform}_item_quantity_sold'] / df[f'{ad_platform}_impressions']).fillna(0)
    df[f'{ad_platform}_conversion_per_spend'] = (df[f'{ad_platform}_conversion'] / (df[f'{ad_platform}_spend'])).fillna(0)
    
    df[f'{ad_platform}_roi'] = (df[f'{ad_platform}_product_revenue'] - df[f'{ad_platform}_spend']) / (df[f'{ad_platform}_spend'] + 1e-9)
    df[f'{ad_platform}_roi'] = df[f'{ad_platform}_roi'].replace([np.inf, -np.inf], 0)

    df[f'all_{ad_platform}_roi'] = (df['all_product_revenue'] - df[f'{ad_platform}_spend']) / (df[f'{ad_platform}_spend'] + 1e-9)
    df[f'all_{ad_platform}_roi'] = df[f'all_{ad_platform}_roi'].replace([np.inf, -np.inf], 0)    
    
    return df


# In[5]:


# def compute_interaction_features(df, ad_platform):
#     """
#     This function derives interactive metrics fom views, impressions, revenue and sale volume values
    
#     Args:
#        df (pd.DataFrame): input dataframe with views, impressions, revenue, spend and sale volume columns
#        ad_platform (str): ad platform, meta and google
#     """
#     df[f'{ad_platform}_views_per_impression'] = (df[f'{ad_platform}_product_detail_views'] / (df[f'{ad_platform}_impressions'] + 1e-9)).fillna(0)
#     df[f'{ad_platform}_clicks_per_impression'] = (df[f'{ad_platform}_clicks'] / (df[f'{ad_platform}_impressions'] + 1e-9)).fillna(0)
#     df[f'{ad_platform}_revenue_per_click'] = (df[f'{ad_platform}_product_revenue'] / (df[f'{ad_platform}_clicks'] + 1e-9)).fillna(0)
#     df[f'{ad_platform}_qty_per_click'] = (df[f'{ad_platform}_quantity_added_to_cart'] / (df[f'{ad_platform}_clicks'] + 1e-9)).fillna(0)
#     df[f'{ad_platform}_roi_per_view'] = (df[f'{ad_platform}_product_revenue'] - df[f'{ad_platform}_spend']) / (df[f'{ad_platform}_product_detail_views'] + 1e-9)
    
#     df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
#     return df


# In[6]:


def engineer_features(df, date_column, date_prefix_list, ad_platform_list):
    """
    This function runs feature engineering methods
    
    Args:
       df (pd.DataFrame): input dataframe to be extened with derived features
       date_column (str): date column name
       prefix_list (str): list of prefixes of date dummies
       ad_platform (str): ad platform, meta and google
    """
    df = encode_date(df, date_column, date_prefix_list)
    df = compute_discount(df)

    for ad_platform in ad_platform_list:
        compute_target_metrics(df, ad_platform)
    
    return df

