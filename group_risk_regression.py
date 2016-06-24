
import pandas as pd
from sklearn import metrics, cross_validation, preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import RandomizedLasso
from tensorflow.contrib import skflow
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings


warnings.filterwarnings("ignore")  # Turn off warnings for now. They are running me crazy
no_of_steps = 10000
lrning_rate = 0.045

file_name = 'datasets/musoni_riskiness_dataset.csv'
file_location = os.getcwd() + '/' + file_name
dataset = pd.read_csv(file_location, low_memory=False)
dataset = dataset.drop(['ID'], axis=1)
target = np.asarray(dataset['Max overdue'], dtype=np.float)
data = np.asarray(dataset.drop(['Max overdue'], axis=1), dtype=np.float)  # All input variables
#  data = np.asarray(dataset['Avg overdue 6m'], dtype=np.float) #  Single input variable
#  xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(data, target, test_size=0.2, random_state=42)
scaler = preprocessing.StandardScaler()


#  calculate the mean square error
def calculate_mse():
    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(data, target, test_size=0.2, random_state=42)
    xtrain = scaler.fit_transform(xtrain)
    regressor = skflow.TensorFlowDNNRegressor(hidden_units=[80, 80], steps=10000, batch_size=1)
    regressor.fit(xtrain, ytrain)
    y_predicted = regressor.predict(scaler.transform(xtest))
    score = metrics.mean_squared_error(y_predicted, ytest)
    score = np.round(score, decimals=3)
    print(score)

#  calculate_mse()


#  Run Principal Component Analysis on the input variables
def make_pca_index():
    scaled_data = scaler.fit_transform(data)
    _pca = PCA(n_components=41)
    _pca.fit(scaled_data)
    #  The amount of varience each principal component accounts for
    var = _pca.explained_variance_ratio_
    var_1 = np.cumsum(np.round(var, decimals=4)*100)
    plt.plot(var_1)
    plt.show()
    #  From above graph, we will take 33 components

#  make_pca_index()


# All input variables
def allvariables_prediction():

    allvariables_dataset = scaler.fit_transform(data)
    regressor = skflow.TensorFlowDNNRegressor(hidden_units=[50, 50], steps=10000, learning_rate=lrning_rate)
    regressor.fit(allvariables_dataset, target)
    y_predicted = regressor.predict(allvariables_dataset)
    return y_predicted


# Predict the outcome based on Days PD today, Bad 6M history, Has 6M history,
# Max days od history, 2m increase, 3m St Dev, # active loans, % savings
def set1_predictions():
    set1_dataset = dataset[['Days PD today', 'Bad 6m history', 'Has 6m history', 'Max days od hist',
                           '2m increase', '3m ST DEV', '# active loans', '% saving']]
    set1_dataset = scaler.fit_transform(set1_dataset)
    regressor = skflow.TensorFlowDNNRegressor(hidden_units=[50, 50], steps=no_of_steps, learning_rate=lrning_rate)
    regressor.fit(set1_dataset, target)
    y_predicted = regressor.predict(set1_dataset)
    return y_predicted


#  Predicting the outcome based on  2m increase, 1m increase, Max days od hist, 3m ST DEV, Bad 6m history, Urban,
# % saving, Avg. balance, Nakuru, # active loans
def set2_prediction():
    set2_dataset = dataset[['2m increase', '1m increase', 'Max days od hist', '3m ST DEV', 'Bad 6m history',
                            'Urban', '% saving', 'Avg. balance', 'Nakuru', '# active loans']]

    set2_dataset = scaler.fit_transform(set2_dataset)
    regressor = skflow.TensorFlowDNNRegressor(hidden_units=[50, 50], steps=no_of_steps, learning_rate=lrning_rate)
    regressor.fit(set2_dataset, target)
    y_predicted = regressor.predict(set2_dataset)
    return y_predicted


#  Select the top features that will maximize output(target)
def select_feature_importance():
    data4columns = dataset.drop(['Max overdue'], axis=1)
    column_names = np.asarray(data4columns.columns.values)
    lasso = RandomizedLasso(alpha=0.025)
    scaled_data = scaler.fit_transform(data)
    lasso.fit(scaled_data, target)
    scores = lasso.scores_
    #  column_names
    #  print scores
    print sorted(zip(map(lambda x: round(x, 4), scores), column_names), reverse=True)

#  select_feature_importance()


# Prediction for selected values
def selectedfeatures_prediction():
    selected_dataset = dataset[['2m increase', 'Bad 6m history', 'Days PD today', 'Avg days od hist',
                                'Max days od hist', '3m ST DEV', '1m increase', 'Max overdue 6m', 'Median 3 m',
                                'Days PD m-1', 'Days PD m-2', 'Avg overdue 6m']]
    selected_dataset = scaler.fit_transform(selected_dataset)
    regressor = skflow.TensorFlowDNNRegressor(hidden_units=[50, 50], steps=no_of_steps, learning_rate=0.045)
    regressor.fit(selected_dataset, target)
    y_predicted = regressor.predict(selected_dataset)
    return y_predicted


# Prediction for the selectedd 26 locations
def top26features_prediction():
    top26_dataset = dataset[['2m increase', 'Bad 6m history', 'Days PD today', 'Avg days od hist',
                             'Max days od hist', '3m ST DEV', '1m increase', 'Max overdue 6m', 'Median 3 m',
                             'Days PD m-1', 'Days PD m-2', 'Avg overdue 6m', 'Urban', 'Avg. principal disbursed',
                             '% female', 'Avg. balance', 'Age at disbursement', 'Active clients', '% saving',
                             '% new clients', '# active loans', 'Loan cycle', 'Has 6m history', '% Wepesi',
                             '% Nawiri', '% KB']]
    top26_dataset = scaler.fit_transform(top26_dataset)
    regressor = skflow.TensorFlowDNNRegressor(hidden_units=[50, 50], steps=no_of_steps, learning_rate=lrning_rate)
    regressor.fit(top26_dataset, target)
    y_predicted = regressor.predict(top26_dataset)
    return y_predicted
dataset['Prediction(set1)'] = set1_predictions()
dataset['Prediction(set2)'] = set2_prediction()
dataset['Prediction(selected)'] = selectedfeatures_prediction()
dataset['Prediction(top 26)'] = top26features_prediction()
dataset['Prediction(all_input_variables)'] = allvariables_prediction()
dataset.to_csv('riskness_predicted_results.csv')
print '-Done-'
