
# Implementing regression version of machine learning

import pandas as pd
from sklearn import metrics, cross_validation, preprocessing
from tensorflow.contrib import skflow
import os
import numpy as np
import warnings


warnings.filterwarnings("ignore")  # Turn off warnings for now. They are running me crazy
file_name = 'datasets/musoni_riskiness_dataset.csv'
file_location = os.getcwd() + '/' + file_name
dataset = pd.read_csv(file_location, low_memory=False)
dataset = dataset.drop(['ID'], axis=1)

target = np.asarray(dataset['Max overdue'], dtype=np.float)

data = np.asarray(dataset.drop(['Max overdue'], axis=1), dtype=np.float)  # All input variables
#  data = np.asarray(dataset['Avg overdue 6m'], dtype=np.float) #  Single input variable
xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(data, target, test_size=0.2, random_state=42)

scaler = preprocessing.StandardScaler()
xtrain = scaler.fit_transform(xtrain)

regressor = skflow.TensorFlowDNNRegressor(hidden_units=[80, 80], steps=10000, batch_size=1)
regressor.fit(xtrain, ytrain)
y_predicted = regressor.predict(scaler.transform(xtest))
score = metrics.mean_squared_error(y_predicted, ytest)
score = np.round(score, decimals=3)
print(score)


