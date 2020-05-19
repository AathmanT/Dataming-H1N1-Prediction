import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


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



##########################################
########FEATURE SELECTION H1N1   #########
##########################################

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
########## H1N1 Train #############
###################################


scaler1 = MinMaxScaler()
# estimator1 = XGBClassifier(penalty="l2", C=100,objective="binary:logistic", random_state=42,min_child_weight= 1, max_depth= 5, learning_rate=0.05, gamma= 5, colsample_bytree= 0.6)
estimator1 = XGBClassifier(penalty="l2", C=100,objective="binary:logistic", random_state=42,learning_rate=0.05)

# features_df = features_df.drop(['census_msa','household_children','hhs_geo_region','household_adults'], axis=1)


# features_h1n1 = features_df.drop(['household_children','hhs_geo_region','household_adults'], axis=1)

features_h1n1 = features_df.iloc[:,:].values
features_h1n1 = scaler1.fit_transform(features_h1n1)

from sklearn.decomposition import KernelPCA
kpca1 = KernelPCA(n_components=2, kernel = 'rbf')
features_h1n1 = kpca1.fit_transform(features_h1n1)
# explained_variance1 = pca1.explained_variance_ratio_

labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# lda1 = LDA(n_components = 3)
# features_h1n1 = lda1.fit_transform(features_h1n1,labels_h1n1)

# explained_variance1 = explained_variance1.reshape(explained_variance1.shape[0],1)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_h1n1,
    labels_h1n1,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)

estimator1.fit(features_h1n1, labels_h1n1)


# test_features_h1n1 = test_features_df.drop(['household_children','hhs_geo_region','household_adults'], axis=1)
test_features_h1n1 = test_features_df.iloc[:, :].values
test_features_h1n1 = scaler1.transform(test_features_h1n1)
test_features_h1n1 = kpca1.transform(test_features_h1n1)
# test_features_h1n1 = lda1.transform(test_features_h1n1)

test_probas1 = estimator1.predict_proba(test_features_h1n1)

# preds = estimator1.predict_proba(X_eval)
#
# y_preds = pd.DataFrame(
#     {
#         "h1n1_vaccine": preds[:, 1],
#     },
#     index = y_eval.index
# )
# print("y_preds.shape:", y_preds.shape)
# y_preds.head()
#
#
# roc_auc_score(y_eval['h1n1_vaccine'], y_preds['h1n1_vaccine'])


##############################################
######## FEATURE SELECTION Seasonal  #########
##############################################

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 10 best features
labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(features_df, labels_seasonal)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features_df.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(35,'Score'))  #print 10 best features


#######################################
########## Seasonal Train #############
#######################################


scaler2 = MinMaxScaler()
# estimator2 = XGBClassifier(penalty="l2", C=100,objective="binary:logistic", random_state=42,min_child_weight= 1, max_depth= 5, learning_rate=0.05, gamma= 5, colsample_bytree= 0.6)
estimator2 = XGBClassifier(penalty="l2", C=100,objective="binary:logistic", random_state=42,learning_rate=0.05)

# features_df = features_df.drop(['census_msa','household_children','hhs_geo_region','household_adults'], axis=1)
# features_seasonal = features_df.drop(['behavioral_antiviral_meds'], axis=1)

features_seasonal = features_df.iloc[:,:].values
features_seasonal = scaler2.fit_transform(features_seasonal)

from sklearn.decomposition import KernelPCA
kpca2 = KernelPCA(n_components=2, kernel = 'rbf')
features_seasonal = kpca2.fit_transform(features_seasonal)
# explained_variance2 = pca2.explained_variance_ratio_
# explained_variance2 = explained_variance2.reshape(explained_variance2.shape[0],1)
labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# lda2 = LDA(n_components=3)
# features_seasonal = lda2.fit_transform(features_seasonal,labels_seasonal)

X_train, X_eval, y_train, y_eval = train_test_split(
    features_seasonal,
    labels_seasonal,
    test_size=0.33,
    shuffle=True,
    stratify=labels_df,
    random_state=RANDOM_SEED
)

estimator2.fit(features_seasonal, labels_seasonal)


# test_features_seasonal = test_features_df.drop(['behavioral_antiviral_meds'], axis=1)
test_features_seasonal = test_features_df.iloc[:, :].values
test_features_seasonal = scaler2.transform(test_features_seasonal)
test_features_seasonal = kpca2.transform(test_features_seasonal)
# test_features_seasonal = lda2.transform(test_features_seasonal)

test_probas2 = estimator2.predict_proba(test_features_seasonal)

# preds = estimator.predict_proba(X_eval)
# preds
#
# y_preds = pd.DataFrame(
#     {
#         "seasonal_vaccine": preds[:, 1],
#     },
#     index = y_eval.index
# )
# print("y_preds.shape:", y_preds.shape)
# y_preds.head()
#
#
# roc_auc_score(y_eval['seasonal_vaccine'], y_preds['seasonal_vaccine'])



# ###################################
# ########## Multi Label  Train ############
# ###################################
# scaler = MinMaxScaler()
#
# features_df = features_df.iloc[:, :].values
# features_df = scaler.fit_transform(features_df)
#
# estimators = MultiOutputClassifier(
#     estimator=XGBClassifier(penalty="l2", C=100, objective="binary:logistic",
#                             random_state=42, learning_rate=0.05,
#                             n_estimators=1000)
# )
#
# X_train, X_eval, y_train, y_eval = train_test_split(
#     features_df,
#     labels_df,
#     test_size=0.33,
#     shuffle=True,
#     stratify=labels_df,
#     random_state=RANDOM_SEED
# )
#
# # Train model
# estimators.fit(features_df, labels_df)
#
# # Predict on evaluation set
# # This competition wants probabilities, not labels
# preds = estimators.predict_proba(X_eval)
# preds
# k = preds[0]
# y_preds = pd.DataFrame(
#     {
#         "h1n1_vaccine": preds[0][:, 1],
#         "seasonal_vaccine": preds[1][:, 1],
#     },
#     index=y_eval.index
# )
# print("y_preds.shape:", y_preds.shape)
# y_preds.head()
#
#
# def plot_roc(y_true, y_score, label_name, ax):
#     fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     ax.plot(fpr, tpr)
#     ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
#     ax.set_ylabel('TPR')
#     ax.set_xlabel('FPR')
#     ax.set_title(
#         f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
#     )
#
#
# fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
#
# plot_roc(
#     y_eval['h1n1_vaccine'],
#     y_preds['h1n1_vaccine'],
#     'h1n1_vaccine',
#     ax=ax[0]
# )
# plot_roc(
#     y_eval['seasonal_vaccine'],
#     y_preds['seasonal_vaccine'],
#     'seasonal_vaccine',
#     ax=ax[1]
# )
# fig.tight_layout()
#
# roc_auc_score(y_eval, y_preds)
# plt.show()


#####################################
########## Multi Label Predict ###############
#####################################

print(test_features_df.shape)

print(submission_df.shape)
print(np.asarray(test_probas1).shape)
print(np.asarray(test_probas2).shape)

submission_df.head()

# Save predictions to submission data frame
submission_df["h1n1_vaccine"] = test_probas1[:, 1]
submission_df["seasonal_vaccine"] = test_probas2[:, 1]

submission_df.head()

submission_df.to_csv('submission8.csv', index=True)



# # Visualising the Training set results H1N1
# from matplotlib.colors import ListedColormap
# X_set, y_set = features_h1n1, labels_h1n1.iloc[:, :].values
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, estimator1.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     X_setttt = X_set[y_set == 0]
#     X_setttt2 = X_set[y_set == 0]
#
#     plt.scatter(X_setttt, X_setttt2,
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('XGBoost (Training set H1N1)')
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.legend()
# plt.show()
#
# # Visualising the Training set results Seasonal
# from matplotlib.colors import ListedColormap
# X_set, y_set = features_seasonal, labels_seasonal
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, estimator2.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('XGBoost (Training set Seasonal)')
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.legend()
# plt.show()