
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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


def check_null_nans(df, variable, input_data_folderpath):
    """
    This function checks the share of missing and zero values in input variables
    """
    print(variable)
    
    null_count = df[variable].isna().sum()
    total_count = len(df)
    null_share = null_count / total_count
    print(f'Nulls in {variable}: {null_count} ({null_share:.2%})')
    
    zero_count = (df[variable] == 0).sum()
    total_count = len(df)
    zero_share = zero_count / total_count
    print(f'Zeros in {variable}: {zero_count} ({zero_share:.2%})')
    
    histogram_filepath = input_data_folderpath + f'\\histograms\\{variable}_hist.jpg'
    
    low, high = df[variable].quantile([0.025, 0.975])
    
    df[variable].hist(bins=1000)
    plt.xlim(low, high)
    plt.title(f'Distribution of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    
    plt.ylim(0, df[variable].value_counts(bins=100).max() * 1.1)
    
    note_nulls = f"Nulls in {variable}: {null_count} ({null_share:.2%})"
    plt.figtext(0.5, -0.05, note_nulls, wrap=True, horizontalalignment='center', fontsize=9)

    note_zeros = f"Zeros in {variable}: {zero_count} ({zero_share:.2%})"
    plt.figtext(0.5, -0.15, note_zeros, wrap=True, horizontalalignment='center', fontsize=9)
   
    plt.tight_layout()
    plt.savefig(histogram_filepath, dpi=300, bbox_inches='tight')
    plt.show()


def check_infs(df, variable):
    """
    This function checks the share of infinite values in derived variables
    """
    print(variable)
    
#     df[variable].hist(bins=50)
#     plt.title(f'Distribution of {variable}')
#     plt.xlabel(variable)
#     plt.ylabel('Frequency')
#     plt.show()
    
    inf_count = np.isinf(df[variable]).sum()
    total_count = len(df)
    inf_share = inf_count/total_count
    print(f'Inf {variable} before replacement: {inf_count} ({inf_share:.2%})')


def compute_target_metrics(df, ad_platform, input_data_folderpath):
    """
    This function derives the target metrics from online perfomance columns by ad platform
    
    Args:
        df (pd.DataFrame): input dataframe with online perfomance columns
        ad_platform (str): ad platform, meta and google
    """
    check_null_nans(df, f'{ad_platform}_spend', input_data_folderpath)
    check_null_nans(df, f'{ad_platform}_impressions', input_data_folderpath)
    check_null_nans(df, f'{ad_platform}_clicks', input_data_folderpath)
    check_null_nans(df, f'{ad_platform}_item_quantity_sold', input_data_folderpath)
    check_null_nans(df, f'{ad_platform}_product_revenue', input_data_folderpath)
     
    df[f'{ad_platform}_impressions_per_spend'] = (df[f'{ad_platform}_impressions'] / (df[f'{ad_platform}_spend'])).fillna(0)
    check_infs(df, f'{ad_platform}_impressions_per_spend')
    df[f'{ad_platform}_impressions_per_spend'] = df[f'{ad_platform}_impressions_per_spend'].replace([np.inf, -np.inf], 0)
    
    df[f'{ad_platform}_clickthrough'] = (df[f'{ad_platform}_clicks'] / df[f'{ad_platform}_impressions']).fillna(0)
    df[f'{ad_platform}_clickthrough_per_spend'] = (df[f'{ad_platform}_clickthrough'] / (df[f'{ad_platform}_spend'])).fillna(0)
    check_infs(df, f'{ad_platform}_clickthrough_per_spend')
    df[f'{ad_platform}_clickthrough_per_spend'] = df[f'{ad_platform}_clickthrough_per_spend'].replace([np.inf, -np.inf], 0)
    
    df[f'{ad_platform}_conversion'] = (df[f'{ad_platform}_item_quantity_sold'] / df[f'{ad_platform}_impressions']).fillna(0)
    df[f'{ad_platform}_conversion_per_spend'] = (df[f'{ad_platform}_conversion'] / (df[f'{ad_platform}_spend'])).fillna(0)
    check_infs(df, f'{ad_platform}_conversion_per_spend')
    df[f'{ad_platform}_conversion_per_spend'] = df[f'{ad_platform}_conversion'].replace([np.inf, -np.inf], 0)
    
    df[f'{ad_platform}_roi'] = (df[f'{ad_platform}_product_revenue'] - df[f'{ad_platform}_spend']) / (df[f'{ad_platform}_spend'] + 1e-9)
    check_infs(df, f'{ad_platform}_roi')
    df[f'{ad_platform}_roi'] = df[f'{ad_platform}_roi'].replace([np.inf, -np.inf], 0)
    
    df[f'all_{ad_platform}_roi'] = (df['all_product_revenue'] - df[f'{ad_platform}_spend']) / (df[f'{ad_platform}_spend'] + 1e-9)
    check_infs(df, f'all_{ad_platform}_roi')
    df[f'all_{ad_platform}_roi'] = df[f'all_{ad_platform}_roi'].replace([np.inf, -np.inf], 0)    

    check_infs(df, f'{ad_platform}_impressions_per_spend')
    check_infs(df, f'{ad_platform}_clickthrough_per_spend')
    check_infs(df, f'{ad_platform}_conversion_per_spend')
    check_infs(df, f'{ad_platform}_roi')
    check_infs(df, f'all_{ad_platform}_roi')
    
    return df



def engineer_features(df, date_column, date_prefix_list, ad_platform_list, input_data_folderpath):
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
        compute_target_metrics(df, ad_platform, input_data_folderpath)
    
    return df

