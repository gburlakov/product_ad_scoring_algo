
import os
import pandas as pd
import numpy as np
import datetime
import pickle

from config import trained_models_folderpath, model_dict


def reverse_one_hot(df, prefix, new_col_name):
    one_hot_cols = [col for col in df.columns if col.startswith(prefix + "_")]
    
    if not one_hot_cols:
        raise ValueError(f"No one-hot columns found with prefix '{prefix}_'")

    extracted = df[one_hot_cols].apply(
        lambda row: next((int(col.split("_")[1]) for col in one_hot_cols if row[col]), np.nan), axis=1
    )

    df[new_col_name] = extracted.astype("Int64")
    
    return df


def predict_metric(trained_models_folderpath, val_last_day_str, X_val, y_val, orig_X_val, target, ad_platform, model_dict):
    """
    Predicts the target values for the validation dataset using trained models.

    Args:
        trained_models_folderpath (str): path to the folder with trained models
        val_last_day_str (str): end date of the validation set
        X_val (pd.DataFrame): validation dataset
        target (str): target metric to be predicted
        target_platform (str): ad platform to predict for, 'meta' or 'google'
    """
    try:
        ensemble_model_filename = os.path.join(trained_models_folderpath, \
                                             f"{val_last_day_str}_{target}_ensemble.pkl")
        with open(ensemble_model_filename, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        encoders_filename = os.path.join(trained_models_folderpath, \
                                                         f"{val_last_day_str}_{target}_encoders.pkl")        
        with open(encoders_filename, 'rb') as f:
            encoders = pickle.load(f)

        scalers_filename = os.path.join(trained_models_folderpath, \
                                                         f"{val_last_day_str}_{target}_scalers.pkl")        
        with open(scalers_filename, 'rb') as f:
            scalers = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading model or preprocessing objects: {e}.  Make sure training has been run.")

    ensemble_weights = ensemble_data['ensemble_weights']
    top_models = ensemble_data['top_models']
    features = ensemble_data['features']

    X = X_val[features]

    for col in X.columns:
        if X[col].dtype == 'object':
            if col in encoders:
                try:
                    X[col] = encoders[col].transform(X[col])
                except ValueError as e:
                    print(f"Warning: Unseen category in column {col}, default value (e.g., -1) assigned. Error: {e}")

                    X[col] = X[col].apply(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
            else:
                print(f"No encoder for column {col}. Encoding skipped.")
                X[col] = 0

    numerical_cols = X.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if col in scalers:
            X[col] = scalers[col].transform(X[[col]])
        else:
            print(f"Warning: No scaler found for column {col}. Skipping scaling.")

    individual_predictions = {}
    for model_name in model_dict.keys():
        model_path = os.path.join(trained_models_folderpath, \
                                                 f"{val_last_day_str}_{target}_{model_name}.pkl")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            individual_predictions[model_name] = model.predict(X).ravel()
        except FileNotFoundError as e:
            print(f"Warning: Model file not found for {model_name}: {e}")
            individual_predictions[model_name] = np.zeros(len(X)).ravel()

    ensemble_predictions = np.zeros(len(X))
    for model_name, model in top_models.items():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            weight = ensemble_weights[model_name]

            ensemble_predictions += weight * model.predict(X).ravel()
        except FileNotFoundError as e:
            print(f"Warning: Model file not found for {model_name}: {e}")
            
    pred_df = pd.DataFrame(individual_predictions, index=X.index)
    pred_df['ensemble'] = ensemble_predictions
    pred_df['actual'] = y_val.values

    merged_df = orig_X_val.copy()
    merged_df[pred_df.columns] = pred_df
    
    pred_cols = list(individual_predictions.keys()) + ['ensemble']

    merged_df.reset_index(inplace=True)
    
    id_vars = [col for col in merged_df.columns
               if col not in pred_cols + ['actual']]

    if 'date' in id_vars:
        id_vars.remove('date')
    id_vars = ['date'] + id_vars
    
    melted_df = merged_df.melt(id_vars=id_vars, 
                               value_vars=list(individual_predictions.keys()) + ['ensemble'], 
                               var_name="model",
                               value_name="pred_value")
    m = len(pred_cols)
    melted_df['actual'] = np.repeat(merged_df['actual'].values, m)

    melted_df['target'] = target
    melted_df['ad_platform'] = ad_platform
    
    if 'date' not in melted_df.columns and 'date' in orig_X_val.columns:
        melted_df['date'] = orig_X_val['date'].values.repeat(m)

    melted_df.reset_index(drop=True, inplace=True)

    sort_cols = ['product_id', 'category', 'model', 'date']
    for col in sort_cols:
        if col not in melted_df.columns:
            print(f"Warning: Column '{col}' not found in melted_df, skipping in sort.")
    melted_df.sort_values(by=[col for col in sort_cols if col in melted_df.columns], inplace=True)

    melted_df.set_index('product_id', inplace=True)
    
    return melted_df

