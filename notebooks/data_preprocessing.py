
import pandas as pd

from config import date_column

def convert_text_to_datetime(df, date_column):
    """
    This function converts date column of a dataframe from text to datetime datatype
    
    Args:
        df (pd.DataFrame): input dataframe with datecolumn
        date_column (str): name of the date column
    """
    df[date_column] = pd.to_datetime(df[date_column])
    
    return df

def impute_missing(df):
    """
    This function imputes mean in numeric columns and mode in the rest if missing values are detected
    
    Args:
       df (pd.DataFrame): input dataframe to be checked for missing values
    """
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def preprocess_data(df, date_column):
    """
    This function runs data cleaning and formatting methods
    
    Args:
       df (pd.DataFrame): input dataframe to be cleaned and reformatted
       date_column (str): name of the date column
    """
    df = convert_text_to_datetime(df, date_column)
    
    return impute_missing(df)

