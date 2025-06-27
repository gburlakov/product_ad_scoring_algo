import pandas as pd
import numpy as np

def rank_products(all_predictions_df, metrics_df, all_ranked_products_filepath):
    """
    This function ranks the product IDs based on predicted values of the target variable
    for each (category, target, ad_platform, model, date) combination.
    """

    df = all_predictions_df.reset_index().rename(columns={'index': 'product_id'})

    if 'date' not in df.columns:
        raise ValueError("'date' column must be present in all_predictions_df")

    ranked_list = []

    combos = df[['category', 'target', 'ad_platform']].drop_duplicates()

    for _, combo in combos.iterrows():
        cat, tgt, plat = combo['category'], combo['target'], combo['ad_platform']

        mask = (
            (metrics_df['category'] == cat) &
            (metrics_df['target'] == tgt) &
            (metrics_df['ad_platform'] == plat) &
            (metrics_df['rank'] == 1)
        )
        best_models = metrics_df.loc[mask, 'model'].unique()
        if len(best_models) == 0:
            continue
        best_model = best_models[0]

        df_sub = df[
            (df['category'] == cat) &
            (df['target'] == tgt) &
            (df['ad_platform'] == plat) &
            (df['model'] == best_model)
        ].copy()

        df_sub['rank'] = df_sub.groupby('date')['pred_value'].rank(ascending=False, method='first')

        ranked_list.append(
            df_sub[['product_id', 'category', 'target', 'ad_platform', 'model', 'date',
                    'pred_value', 'actual', 'rank']]
        )

    all_ranked_products_df = pd.concat(ranked_list, ignore_index=True)

    all_ranked_products_df.sort_values(
        by=['category', 'target', 'ad_platform', 'model', 'date', 'rank'],
        inplace=True
    )

    all_ranked_products_df.to_csv(all_ranked_products_filepath, index=False)

    return all_ranked_products_df
