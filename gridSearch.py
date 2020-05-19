import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from gsmote.oldgsmote import OldGeometricSMOTE
# from gsmote.eg_smote import EGSmote
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# train_filename = "Data/train.csv"
# df_train = pd.read_csv(train_filename)
# X = pp.preProcess_X(df_train)
# y = pp.preProcess_y(df_train)



features_df = pd.read_csv("training_set_features.csv", index_col="respondent_id")

labels_df = pd.read_csv("training_set_labels.csv", index_col="respondent_id")

test_features_df = pd.read_csv( "test_set_features.csv", index_col="respondent_id")

submission_df = pd.read_csv("submission_format.csv", index_col="respondent_id")


####################
le = LabelEncoder()

###############################
### PreProcess Train data #####
###############################
print(features_df.columns[features_df.isnull().any()])

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
features_df = features_df.drop(['census_msa'], axis=1)

#################################
##### PreProcess test data ######
#################################

test_features_df.fillna(test_features_df.mode().iloc[0], inplace=True)
test_features_df['education'] = le.fit_transform(test_features_df['education'])
test_features_df['age_group'] = le.fit_transform(test_features_df['age_group'])
test_features_df['race'] = le.fit_transform(test_features_df['race'])
test_features_df['sex'] = le.fit_transform(test_features_df['sex'])
test_features_df['income_poverty'] = le.fit_transform(test_features_df['income_poverty'])
test_features_df['marital_status'] = le.fit_transform(test_features_df['marital_status'])
test_features_df['rent_or_own'] = le.fit_transform(test_features_df['rent_or_own'])
test_features_df['employment_status'] = le.fit_transform(test_features_df['employment_status'])
test_features_df['hhs_geo_region'] = le.fit_transform(test_features_df['hhs_geo_region'])
test_features_df['census_msa'] = le.fit_transform(test_features_df['census_msa'])
test_features_df['employment_industry'] = le.fit_transform(test_features_df['employment_industry'])
test_features_df['employment_occupation'] = le.fit_transform(test_features_df['employment_occupation'])
test_features_df = test_features_df.drop(['census_msa'], axis=1)

# features_df = features_df.drop(['census_msa','household_children','hhs_geo_region','household_adults'], axis=1)
features_h1n1 = features_df.drop(['household_children','hhs_geo_region','household_adults'], axis=1)

features_h1n1 = features_h1n1.iloc[:,:].values

labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features_h1n1, labels_h1n1, test_size=0.2, random_state=0)

# sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
# sm = OldGeometricSMOTE()
# sm = EGSmote()
# X_train, y_train = sm.fit_resample(X_train, y_train)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

params = {
        'min_child_weight': [1, 2, 5],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6, 7]
        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
# sm = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
# sm = OldGeometricSMOTE()
# sm = EGSmote()
# X, y = sm.fit_resample(X, y)
folds = 3
param_comb = 600

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=-1, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(features_h1n1, labels_h1n1)
timer(start_time) # timing ends here for "start_time" variable

# print('\n All results:')
# print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)