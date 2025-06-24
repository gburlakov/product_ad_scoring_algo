#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import pandas as pd


# In[14]:


from config import input_data_xlsx_filepath, input_data_csv_filepath


# In[15]:


def load_data(input_data_xlsx_filepath, input_data_csv_filepath):
    """
    This function loads the input data from an excel file and converts it to a .csv file
    
    Args:
        input_data_xlsx_filepath (str): input .xlsx data file
        input_data_csv_filepath (str): input .csv data file
        
    """
    df = pd.read_excel(input_data_xlsx_filepath, engine='openpyxl')
    df.to_csv(input_data_csv_filepath, index=False)


# In[16]:


input_df = pd.read_csv(input_data_csv_filepath)
input_df.info()


# In[20]:


input_df['date'] = pd.to_datetime(input_df['date'])
val_last_day = input_df['date'].max()
val_first_date = val_last_day - pd.Timedelta(days=7)


# In[21]:


val_first_date


# In[22]:


val_last_day


# In[24]:


input_df[(input_df['date']>=val_first_date)&(input_df['date']<=val_last_day)]

