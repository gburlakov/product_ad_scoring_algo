#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


from config import date_column


# In[4]:


def convert_text_to_datetime(df, date_column):
    """
    This function converts date column of a dataframe from text to datetime datatype
    
    Args:
        df (pd.DataFrame): input dataframe with datecolumn
        date_column (str): name of the date column
    """
    df[date_column] = pd.to_datetime(df[date_column])
    
    return df


# In[5]:


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


# In[6]:


def preprocess_data(df, date_column):
    """
    This function runs data cleaning and formatting methods
    
    Args:
       df (pd.DataFrame): input dataframe to be cleaned and reformatted
       date_column (str): name of the date column
    """
    df = convert_text_to_datetime(df, date_column)
    
    return impute_missing(df)

