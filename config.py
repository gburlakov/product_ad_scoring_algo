
import os
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor, \
ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


cwd=os.getcwd()
cwd = os.path.dirname(cwd)


data_folderpath = cwd + '\product_ad_scoring_algo\\data'

input_data_folderpath = data_folderpath + '\input'

xlsx_filename = 'data_scientist_product_data.xlsx'
csv_filename = xlsx_filename[:-5] + '.csv'

input_data_xlsx_filepath = input_data_folderpath + '\\' + xlsx_filename
input_data_csv_filepath = input_data_folderpath + '\\' + csv_filename

meta_data_folderpath = data_folderpath + '\\meta\\'
trained_models_folderpath = meta_data_folderpath + 'trained_models\\'

output_data_folderpath = data_folderpath + '\\output\\'
predictions_folderpath = output_data_folderpath + '\\predictions\\'

all_pred_csv_filename = 'all_pred.csv'
all_predictions_csv_filepath = predictions_folderpath + '\\' + all_pred_csv_filename

all_validation_metrics_filename = 'all_validation_metrics.csv'
all_validation_metrics_filepath = predictions_folderpath + '//' + all_validation_metrics_filename

scoring_folderpath = output_data_folderpath + '\\ranking\\'

all_ranked_products_filename = 'all_ranked_products.csv'
all_ranked_products_filepath = scoring_folderpath + '\\' + all_ranked_products_filename

ABtest_folderpath = output_data_folderpath + '\\ABtesting\\'

all_actual_products_selected_filename = 'all_actual_products_selected.csv'
all_actual_products_selected_filepath = ABtest_folderpath + all_actual_products_selected_filename

all_pred_products_selected_filename = 'all_pred_products_selected.csv'
all_pred_products_selected_filepath = ABtest_folderpath + all_pred_products_selected_filename 

all_ab_test_results_filename = 'all_ab_test_results.csv'
all_ab_test_results_filepath = ABtest_folderpath + all_ab_test_results_filename

date_column = 'date'
date_prefix_list = ['yr', 'm', 'w', 'wd']

ad_platform_list = ['meta', 'google']

train_size = 0.7
test_size = 0.15
val_size = 0.15

target_dict = {ad_platform: [
    f'{ad_platform}_roi'
] for ad_platform in ad_platform_list}


# target_dict = {ad_platform: [
#     f'{ad_platform}_impressions_per_spend',
#     f'{ad_platform}_clickthrough_per_spend',
#     f'{ad_platform}_conversion_per_spend',
#     f'{ad_platform}_roi',
#     f'all_{ad_platform}_roi'
# ] for ad_platform in ad_platform_list}


feature_list = [
    'category',
    'sale_price',
    'discount_rate',
    'product_age',
    'product_status',
    'pct_product_variants_in_stock',
]

model_dict = {
    'LinearRegression': (LinearRegression(), {}),
    'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [None, 5, 10]}),
    'KNeighbors': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]})
}


# MODELS = {
#     'GradientBoosting': (GradientBoostingRegressor(), {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}),
#     'RandomForest': (RandomForestRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}),
#     'HistGradientBoosting': (HistGradientBoostingRegressor(), {'learning_rate': [0.01, 0.05, 0.1], 'max_iter': [100, 200, 300], 'max_depth': [None, 3, 5]}),
#     'ExtraTrees': (ExtraTreesRegressor(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}),
#     'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.05, 0.1]}),
#     'LinearRegression': (LinearRegression(), {}),
#     'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [None, 5, 10]}),
#     'KNeighbors': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
#     'SVR': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
# }


n_trials_optuna = 1


def save_val_end_points(df):
    """
    This function takes the last day of the dataset and take one week back as end points of the validation set 
    """
    val_last_day = df['date'].max()
    val_first_day = val_last_day - pd.Timedelta(days=7)
    print(type(val_first_day))
    
    val_last_day_str = val_last_day.strftime('%Y-%m-%d')
    
    return val_first_day, val_last_day, val_last_day_str


# for ad_platform in ad_platform_list:
#     model_dict = {
#         ad_platform: {'ensemble': [meta_data_folderpath + '\\' + f'{ad_platform}_ensemble_model.pkl'], 'encoders': [meta_data_folderpath + '\\' + f'{ad_platform}_encoders.pkl'], 'scalers': [meta_data_folderpath + '\\' + f'{ad_platform}_scalers.pkl']}
#         for ad_platform in ad_platform_list
#     }

