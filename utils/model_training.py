
import os

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import optuna

from config import input_data_csv_filepath, ad_platform_list, train_size, model_dict, \
target_dict, feature_list, trained_models_folderpath, n_trials_optuna


def set_train_test_val_size(df, val_first_day, val_last_day, train_size, target, features, date_column, prefixes, dummy_columns, \
                            encoding=True):
    """
    Splits the input dataset into train, test, and validation sets.

    Args:
        df (pd.DataFrame): DataFrame containing processed product data.
        val_first_day (str or pd.Timestamp): First day of the validation period.
        val_last_day (str or pd.Timestamp): Last day of the validation period.
        train_size (float): Share of data to use for training.
        target (str): Target metric to be predicted.
        features (list): List of features to train models on.
        date_column (str): Name of the date column.
        encoding (bool): Whether to apply encoding and scaling to features.
    
    Returns:
        Depending on `encoding`, returns either:
            - Encoded version: X_train, X_test, X_val, y_train, y_test, y_val, encoders, scalers
            - Raw version: X_val, y_val
    """
    X = df[features + [date_column]].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[[target, date_column]]

    encoders, scalers = {}, {}

    if encoding:
        for col in features:
            if X[col].dtype == 'object':
                encoders[col] = LabelEncoder()
                X[col] = encoders[col].fit_transform(X[col])
        
        numerical_cols = X[features].select_dtypes(include=np.number).columns
        for col in numerical_cols:
            scalers[col] = StandardScaler()
            X[col] = scalers[col].fit_transform(X[[col]])

    mask_val = (X[date_column] >= val_first_day) & (X[date_column] <= val_last_day)
    mask_train_test = X[date_column] < val_first_day

    X_val = X[mask_val].copy()
    y_val = y[mask_val].copy()

    X_train_test = X[mask_train_test].copy()
    y_train_test = y[mask_train_test].copy()

    drop_date = encoding
    if drop_date:
        X_val.drop(columns=[date_column], inplace=True)
        y_val.drop(columns=[date_column], inplace=True)
        X_train_test.drop(columns=[date_column], inplace=True)
        y_train_test.drop(columns=[date_column], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test, y_train_test, train_size=train_size, random_state=42
    )

    if encoding:
        return X_train, X_test, X_val, y_train, y_test, y_val, encoders, scalers
    else:
        X_val = X_val.drop(columns=dummy_columns)
        print(X_val.info())
        return X_val, y_val


# In[5]:


def objective(trial, model_class, param_grid, X_train, y_train, X_test, y_test):
    """
    Objective function for Optuna optimization
    
    Args:
        trial (optuna.Trial): Optuna trial object
        model_class (class): ML method
        param_grid (grid): grid with model hyperparameter value sets
        model_name (str): name of the model being optimized
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training target
        X_test (pd.DataFrame): testing features
        y_test (pd.Series): testing target
    """
    params = {}
    for param_name, param_values in param_grid.items():
        if isinstance(param_values, list):
            if all(isinstance(val, (int, float)) for val in param_values):
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                else:
                    params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))

            else:
                params[param_name] = trial.suggest_categorical(param_name, param_values)
 
        else:
            params = {}

    model = model_class.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse


# In[6]:


def train_model(df, ad_platform, target, features, val_last_day_str, trained_models_folderpath, \
                X_train, X_test, y_train, y_test, X_val, y_val, encoders, scalers, model_dict, n_trials_optuna):
    """
    Trains an ensemble of machine learning models for ROI prediction

    Args:
        df (pd.DataFrame): DataFrame containing processed product data
        ad_platform (str): ad platform, meta or google
        target (str): target metric to be trained on
        features (list): list of features to train models on
        val_last_day_str (str): last date of the validation set as a string
        trained_models_folderpath (path): location path of the trained models
        X_train (pd.DataFrame): X features of the train set
        X_test (pd.DataFrame): X features of the test set
        y_train (pd.DataFrame): target y values of the train set
        y_test (pd.DataFrame): target y values of the test set
        X_val (pd.DataFrame): X features of the validation set
        y_val (pd.DataFrame): target y values of the validation set
        encoders: encoder labels of columns
        scalers: standard scaling of the numeric columns values
        model_dict (dict): dictionary with trained model_names, models and parameter grids
    """
    best_models = {}

    for model_name, (model_class, param_grid) in model_dict.items():
        print(f"Optimizing {model_name}...")

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, model_class, param_grid, X_train, y_train, X_test, y_test), n_trials=n_trials_optuna)

        best_params = study.best_params
        model = model_class.set_params(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        best_models[model_name] = {'model': model, 'mse': mse, 'r2': r2}
        print(f"{model_name} - Best MSE: {mse}, R2: {r2}, Best Params: {best_params}")
        
        individual_model_filename = os.path.join(trained_models_folderpath, \
                                                 f"{val_last_day_str}_{target}_{model_name}.pkl")
        with open(individual_model_filename, 'wb') as f:
            pickle.dump(model, f)        
        
    top_3_models = sorted(best_models.items(), key=lambda item: item[1]['r2'], reverse=True)[:3]
    print("\nTop 3 Models:")
    for model_name, model_data in top_3_models:
        print(f"{model_name}: MSE = {model_data['mse']}, R2 = {model_data['r2']}")

    total_r2 = sum(model_data['r2'] for model_name, model_data in top_3_models)
    ensemble_weights = {model_name: model_data['r2'] / total_r2 for model_name, model_data in top_3_models}
    print("\nEnsemble Weights:")
    for model_name, weight in ensemble_weights.items():
        print(f"{model_name}: {weight}")

    # Save the ensemble weights, top models, encoders, and scalers
    ensemble_data = {
        'ensemble_weights': ensemble_weights,
        'top_models': {model_name: os.path.join(trained_models_folderpath, \
                                                f"{val_last_day_str}_{target}_{model_name}.pkl") \
                       for model_name, model_data in top_3_models},
        'features': features
    }
    
    ensemble_model_filename = os.path.join(trained_models_folderpath, \
                                                 f"{val_last_day_str}_{target}_ensemble.pkl")
    with open(ensemble_model_filename, 'wb') as f:
        pickle.dump(ensemble_data, f)
    
    encoders_filename = os.path.join(trained_models_folderpath, \
                                                     f"{val_last_day_str}_{target}_encoders.pkl")    
    with open(encoders_filename, 'wb') as f:
        pickle.dump(encoders, f)
    
    scalers_filename = os.path.join(trained_models_folderpath, \
                                                     f"{val_last_day_str}_{target}_scalers.pkl")    
    with open(scalers_filename, 'wb') as f:
        pickle.dump(scalers, f)

