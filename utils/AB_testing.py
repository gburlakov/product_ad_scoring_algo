
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


def ab_test_selection(all_ranked_products_df, all_predictions_df,
                      all_actual_products_selected_filepath, all_pred_products_selected_filepath, all_ab_test_results_filepath):
    """
    This function runs an A/B testing assessment of the significance of the difference between average ROI
    from the actual product selection and the model implied product selection.
    """
    temp = all_predictions_df.copy()

    if 'product_id' not in temp.columns:
        temp = temp.reset_index().rename(columns={'index': 'product_id'})

    prediction_cols_to_keep = ['product_id', 'model', 'date', 'product_status']
    prediction_cols_to_keep = [col for col in prediction_cols_to_keep if col in temp.columns]

    merge_keys = ['product_id']
    for col in ['model', 'date']:
        if col in all_ranked_products_df.columns and col in temp.columns:
            merge_keys.append(col)

    ranked = all_ranked_products_df.merge(
        temp[prediction_cols_to_keep],
        on=merge_keys,
        how='left'
    )
    
    actual_list = []
    pred_list = []
    ab_list = []

    combos = ranked[['category', 'target', 'ad_platform']].drop_duplicates()

    for _, combo in combos.iterrows():
        cat, tgt, plat = combo['category'], combo['target'], combo['ad_platform']
        subset = ranked[
            (ranked['category'] == cat) &
            (ranked['target'] == tgt) &
            (ranked['ad_platform'] == plat)
        ]

        actual_df = subset[subset['product_status'] != 'DELETED']

        L = len(actual_df)
        pred_df = subset.nsmallest(L, 'rank')

        actual_list.append(actual_df.copy())
        pred_list.append(pred_df.copy())
        
        mean_actual = actual_df['actual'].mean()
        mean_pred = pred_df['actual'].mean()

        t_stat, p_val = ttest_ind(pred_df['actual'], actual_df['actual'], equal_var=False)

        ab_results = pd.DataFrame({
            'category': cat,
            'target': tgt,
            'ad_platform': plat,
            'test_statistic': t_stat,
            'p_value': p_val,
            'mean_actual': mean_actual,
            'mean_predicted': mean_pred
        }, index=[0])
        ab_list.append(ab_results)

    all_actual_products_selected_df = pd.concat(actual_list, ignore_index=True).drop_duplicates()
    all_pred_products_selected_df   = pd.concat(pred_list,   ignore_index=True).drop_duplicates()
    all_ab_test_results_df          = pd.concat(ab_list,     ignore_index=True)

    sort_cols = ['product_id', 'category', 'model', 'date']
    for df in [all_actual_products_selected_df, all_pred_products_selected_df]:
        df.sort_values(by=[col for col in sort_cols if col in df.columns], inplace=True)

    all_actual_products_selected_df.to_csv(all_actual_products_selected_filepath, index=False)
    all_pred_products_selected_df.to_csv(all_pred_products_selected_filepath, index=False)
    all_ab_test_results_df.to_csv(all_ab_test_results_filepath, index=False)

    return all_actual_products_selected_df, all_pred_products_selected_df, all_ab_test_results_df