import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import xgboost
import shap

# load JS visualization code to notebook
from sklearn.preprocessing import LabelEncoder

shap.initjs()



features_df = pd.read_csv("training_set_features.csv", index_col="respondent_id")

labels_df = pd.read_csv("training_set_labels.csv", index_col="respondent_id")


le = LabelEncoder()

features_df.fillna(features_df.mode().iloc[0], inplace=True)
features_df['education'] = le.fit_transform(features_df['education'])
features_df['age_group'] = le.fit_transform(features_df['age_group'])
features_df['race'] = le.fit_transform(features_df['race'])
features_df['sex'] = le.fit_transform(features_df['sex'])
features_df['income_poverty'] = le.fit_transform(features_df['income_poverty'])
features_df['marital_status'] = le.fit_transform(features_df['marital_status'])
features_df['rent_or_own'] = le.fit_transform(features_df['rent_or_own'])
features_df['employment_status'] = le.fit_transform(features_df['employment_status'])
features_df['hhs_geo_region'] = le.fit_transform(features_df['hhs_geo_region'])
features_df['census_msa'] = le.fit_transform(features_df['census_msa'])
features_df['employment_industry'] = le.fit_transform(features_df['employment_industry'])
features_df['employment_occupation'] = le.fit_transform(features_df['employment_occupation'])


labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)
labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)


model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(features_df.values, label=labels_seasonal.values), 100)



# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features_df)
shap.summary_plot(shap_values, features_df, plot_type="bar")