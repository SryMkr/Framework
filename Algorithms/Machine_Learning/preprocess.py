import pandas as pd

# define the dataset column
column_names = ['Session', 'PhoLength', 'CorLength', 'Wrong', 'Correct', 'Accuracy', 'Contribution']
# read the dataset
data_df = pd.read_csv('record.csv', header=None, names=column_names)

# define the features and label
feature_columns = ['Session', 'PhoLength', 'CorLength', 'Wrong', 'Correct', 'Accuracy']
label_column = 'Contribution'

# group by features and calculate the mean of label
data_df_mean = data_df.groupby(feature_columns)[label_column].sum().reset_index()
data_df_mean['Contribution'] = data_df['Contribution'].round(3)
# save to the new csv file
data_df_mean.to_csv('dataset.csv', index=False)


