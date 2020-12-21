import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LassoCV
import pandas as pd
from sklearn.preprocessing import LabelEncoder



features_df = pd.read_csv("training_set_features.csv", index_col="respondent_id")

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

X = features_df

labels_df = pd.read_csv("training_set_labels.csv", index_col="respondent_id")
labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)
labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)
y = labels_seasonal


reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model Seasonal")
plt.show()