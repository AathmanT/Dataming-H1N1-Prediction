from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from yellowbrick.target import FeatureCorrelation
import numpy as np
import pandas as pd

# Load the regression dataset
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


labels_df = pd.read_csv("training_set_labels.csv", index_col="respondent_id")
labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)
labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)

# Create a list of the feature names
features = np.array(features_df.columns)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

visualizer.fit(features_df.values, labels_h1n1.values)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure

# from sklearn import datasets
# from yellowbrick.target import FeatureCorrelation
# import numpy as np
#
#
# # Load the regression dataset
# data = datasets.load_diabetes()
# X, y = data['data'], data['target']
#
# # Create a list of the feature names
# features = np.array(data['feature_names'])
#
# # Instantiate the visualizer
# visualizer = FeatureCorrelation(labels=features)
#
# visualizer.fit(X, y)        # Fit the data to the visualizer
# visualizer.show()           # Finalize and render the figure