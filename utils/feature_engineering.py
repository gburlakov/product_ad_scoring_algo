
import pandas as pd
import numpy as np

from config import date_column, date_prefix_list, ad_platform_list


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

