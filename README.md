# product_ad_scoring_algo
This project aims at developing a scoring algorithm that helps decide which products should be prioritized for advertising across ad channels like Meta and Google.

Key modules in the proposed solution are:

- data load.py
  converting the raw .xlsx file to quicker for loading.csv file  
- data_preprocessing.py
  cleaning NaNs and formatting date column to datetime date type
- feature_engineering.py
  generating TS dummies and target metrics 
- model_training.py
  splitting data into train, test and validation sets, setting target and feature variables, training and test validating set of models
- metrics_prediction.py
  making prediction on the validation set
- model_revalidation.py
  validating best model by product category on the validation set
- product_scoring.py
  ranking products by target metric value predicted with the best revalidated model
- AB_testing.py
  truncating ranked list of product IDs to fit the length of the actual product IDs advertised online (with no DELETED status),
  using AB (t-)testing to estimate the significance of the difference in mean target metric values of the actual and predicted product IDs selected  

Here's the file structure of the code:

<pre> """text C:.
│   .gitignore
│   config.py
│   environment.yml
│   main.py
│   README.md
│   requirements.txt
│
├───data
│   ├───input
│   │       data_scientist_product_data.csv
│   │       data_scientist_product_data.xlsx
│   │
│   ├───meta
│   │   │   data_scientist_column_metadata.txt
│   │   │
│   │   └───trained_models
│   │           2025-05-31_google_roi_DecisionTree.pkl
│   │           2025-05-31_google_roi_encoders.pkl
│   │           2025-05-31_google_roi_ensemble.pkl
│   │           2025-05-31_google_roi_KNeighbors.pkl
│   │           2025-05-31_google_roi_LinearRegression.pkl
│   │           2025-05-31_google_roi_scalers.pkl
│   │           2025-05-31_meta_roi_DecisionTree.pkl
│   │           2025-05-31_meta_roi_encoders.pkl
│   │           2025-05-31_meta_roi_ensemble.pkl
│   │           2025-05-31_meta_roi_KNeighbors.pkl
│   │           2025-05-31_meta_roi_LinearRegression.pkl
│   │           2025-05-31_meta_roi_scalers.pkl
│   │
│   └───output
│       ├───ABtesting
│       │       all_ab_test_results.csv
│       │       all_actual_products_selected.csv
│       │       all_pred_products_selected.csv
│       │
│       ├───predictions
│       │       all_pred.csv
│       │
│       └───ranking
│               all_ranked_products.csv
│
│  
├───utils
│     AB_testing.py
│     data_load.py
│     data_preprocessing.py
│     feature_engineering.py
│     metrics_prediction.py
│     model_revalidation.py
│     model_training.py
│     product_scoring.py """<\pre>