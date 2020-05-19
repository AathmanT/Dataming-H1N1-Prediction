# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:15:45 2020

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

##########################
##### LOAD DATA  #########
##########################
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


#############################
######## Print shape ########
#############################

print(test_features_df.shape)
print(submission_df.shape)

print(features_df.shape)
print(labels_df.shape)


print(features_df.isnull().sum())
print(test_features_df.isnull().sum())

#############################


features_df = features_df.iloc[:,:]
print(features_df.shape)

test_features_df = test_features_df.iloc[:,:]

print("labels_df.shape", labels_df.shape)
labels_df.head()

# np.testing.assert_array_equal(features_df.index.values, labels_df.index.values)

# Phi Coefficient is the same as Pearson for two binary variables
(labels_df["h1n1_vaccine"].corr(labels_df["seasonal_vaccine"], method="pearson"))


#############################
##### BUILDING MODELS #######
#############################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, roc_auc_score

RANDOM_SEED = 6    # Set a random seed for reproducibility!

#####features_df.dtypes != "object"

###################################
########FEATURE SELECTION #########
###################################
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 10 best features
labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(features_df, labels_h1n1)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features_df.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(35,'Score'))  #print 10 best features


###################################
########### H1N1 ##################
###################################
scaler = MinMaxScaler()
estimator=XGBClassifier(penalty="l2", C=100,objective="binary:logistic", random_state=42, learning_rate = 0.05)
# features_df = features_df.drop(['census_msa','household_children','hhs_geo_region','household_adults'], axis=1)
features_df = features_df.drop(['household_children','hhs_geo_region','household_adults'], axis=1)

features_df = features_df.iloc[:,:].values
features_df = scaler.fit_transform(features_df)

labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_h1n1,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)

estimator.fit(X_train, y_train)

preds = estimator.predict_proba(X_eval)
preds

y_preds = pd.DataFrame(
    {
        "h1n1_vaccine": preds[:, 1],
    },
    index = y_eval.index
)
print("y_preds.shape:", y_preds.shape)
y_preds.head()


roc_auc_score(y_eval['h1n1_vaccine'], y_preds['h1n1_vaccine'])

###################################
########## Multi Label  Train ############
###################################
scaler = MinMaxScaler()

features_df = features_df.iloc[:,:].values
features_df = scaler.fit_transform(features_df)

estimators = MultiOutputClassifier(
    estimator=XGBClassifier(penalty="l2", C=100,objective="binary:logistic", 
                            random_state=42, learning_rate = 0.05,
                            n_estimators=1000)
)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_df,
    labels_df,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)


# Train model
estimators.fit(features_df, labels_df)

# Predict on evaluation set
# This competition wants probabilities, not labels
preds = estimators.predict_proba(X_eval)
preds
k = preds[0]
y_preds = pd.DataFrame(
    {
        "h1n1_vaccine": preds[0][:, 1],
        "seasonal_vaccine": preds[1][:, 1],
    },
    index = y_eval.index
)
print("y_preds.shape:", y_preds.shape)
y_preds.head()


def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(
        f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
    )
    
    
fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))

plot_roc(
    y_eval['h1n1_vaccine'], 
    y_preds['h1n1_vaccine'], 
    'h1n1_vaccine',
    ax=ax[0]
)
plot_roc(
    y_eval['seasonal_vaccine'], 
    y_preds['seasonal_vaccine'], 
    'seasonal_vaccine',
    ax=ax[1]
)
fig.tight_layout()


roc_auc_score(y_eval, y_preds)

#####################################
########## Multi Label Predict ###############
#####################################

test_features_df = test_features_df.iloc[:,:].values
test_features_df = scaler.fit_transform(test_features_df)

test_probas = estimators.predict_proba(test_features_df)
test_probas

print(test_features_df.shape)

print(submission_df.shape)
print(np.asarray(test_probas).shape)
submission_df.head()

# Save predictions to submission data frame
submission_df["h1n1_vaccine"] = test_probas[0][:, 1]
submission_df["seasonal_vaccine"] = test_probas[1][:, 1]

submission_df.head()

submission_df.to_csv('submission8.csv', index=True)