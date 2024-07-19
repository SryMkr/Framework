import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# read dataset
dataset_df = pd.read_csv('dataset.csv')

# define the features and label
feature_columns = ['Session', 'PhoLength', 'CorLength', 'Wrong', 'Correct', 'Accuracy']
label_column = 'Contribution'

# split the features and label
features = dataset_df[feature_columns]
labels = dataset_df[label_column]

# split the train and test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 归一化
scaler = MinMaxScaler()

# scale the training dataset
X_train = scaler.fit_transform(X_train)

# scale the testing dataset
X_test = scaler.transform(X_test)

# formulate the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# train model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=False)

# save model
model_filename = 'model.h5'
model.save(model_filename)

# save scaler
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)

# 使用加载的模型进行预测
y_predictions = model.predict(X_test, verbose=0)
print(y_test)
print(y_predictions)

