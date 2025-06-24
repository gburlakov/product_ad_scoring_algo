#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle


# In[ ]:


from config import model_dict, n_trials_optuna


# In[2]:


def predict_roi(X_val, model_dict, ad_platform):
    """
    Predicts the ROI for the validation dataset using the trained ensemble model.

    Args:
        X_val (pd.DataFrame): validation dataset
        model_dict (dict): dictionary with names of .pkl files where trained models, encoders an scalers are stored
        target_platform (str): ad platform to predict for, 'meta' or 'google'
    """
    try:
        with open(model_dict[ad_platform]['ensemble'][0], 'rb') as f:
            ensemble_data = pickle.load(f)
        with open(model_dict[ad_platform]['encoders'][0], 'rb') as f:
            encoders = pickle.load(f)
        with open(model_dict[ad_platform]['scalers'][0], 'rb') as f:
            scalers = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading model or preprocessing objects: {e}.  Make sure training has been run.")

    # Extract ensemble components
    ensemble_weights = ensemble_data['ensemble_weights']
    top_models = ensemble_data['top_models']
    features = ensemble_data['features']

    X = X_val[features]

    # Encode categorical features using the *same* encoders fit during training
    for col in X.columns:
        if X[col].dtype == 'object':
            if col in encoders:  # Check if encoder exists (handle unseen categories)
                try:
                    X[col] = encoders[col].transform(X[col])
                except ValueError as e:
                    print(f"Warning: Unseen category in column {col}, default value (e.g., -1) assigned. Error: {e}")

                    X[col] = X[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
            else:
                print(f"No encoder for column {col}. Encoding skipped.")
                X[col] = 0

    # Scale numerical features using the *same* scalers fit during training
    numerical_cols = X.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if col in scalers: #Check if scaler exists
            X[col] = scalers[col].transform(X[[col]])
        else:
            print(f"Warning: No scaler found for column {col}. Skipping scaling.")

    individual_predictions = {}
    for model_name, model_path in top_models.items():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            individual_predictions[model_name] = model.predict(X)
        except FileNotFoundError as e:
            print(f"Warning: Model file not found for {model_name}: {e}")
            individual_predictions[model_name] = np.zeros(len(X))

    ensemble_predictions = np.zeros(len(X))
    for model_name, model in top_models.items():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            weight = ensemble_weights[model_name]
            ensemble_predictions += weight * model.predict(X)
        except FileNotFoundError as e:
            print(f"Warning: Model file not found for {model_name}: {e}")
            
    predictions_df = pd.DataFrame(individual_predictions, index=X.index)
    predictions_df['ensemble_prediction'] = ensemble_predictions
    
    return predictions_df

