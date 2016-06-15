
# Implementing the classification version of machine learning
# Note  Ensure that all the input variables are standardized

import pandas as pd
from sklearn import preprocessing
import os
file_name = 'datasets/musoni_group_dataset.csv'
file_location = os.getcwd() + '/' + file_name
dataset = pd.read_csv(file_location, low_memory=False)
dataset = dataset.drop(['ID'], axis=1)
print dataset.info()
target = dataset['Max overdue']
# For now, we are loading Avg. balance, Avg. principal disbursed , # active loans, Active clients
# Loan cycle, % female, Age at disbursement, Urban
data = dataset.iloc[:, [9, 10, 11, 12, 13, 14, 15, 35]]
data['% female'] = data['% female'].map(lambda x: str(x)[:-1])  # Remove the percentage sign
# Scale the data
std_scale = preprocessing.StandardScaler().fit(data)
scaled_data = std_scale.transform(data)

