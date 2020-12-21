import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


features_df = pd.read_csv("training_set_features.csv", index_col="respondent_id")


corrMatrix = features_df.corr()
fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches

sn.heatmap(corrMatrix, linewidths=.5,ax=ax,annot=False)
plt.show()




from speedml import Speedml

features_df = pd.read_csv("training_set_features.csv", index_col="respondent_id")


labels_df = pd.read_csv("training_set_labels.csv", index_col="respondent_id")
labels_h1n1 = labels_df.drop(['seasonal_vaccine'], axis=1)
labels_seasonal = labels_df.drop(['h1n1_vaccine'], axis=1)

result = pd.concat([features_df, labels_h1n1], axis=1)
result.to_csv("h1n1_train_and_labels.csv", encoding='utf-8')


sml = Speedml('h1n1_train_and_labels.csv',
              target = 'h1n1_vaccine', uid = 'respondent_id')

sml.shape()