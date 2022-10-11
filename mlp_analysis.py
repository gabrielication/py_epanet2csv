from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = '1M_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'

print("Loading csv...")

# hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,
# has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand

data = pd.read_csv(filename)

print("Dividing X and y matrices...")

X = data.copy().drop(columns=["hour","nodeID","node_type","has_leak","demand_value"])
y = data["demand_value"].copy()

# MLP

le = preprocessing.LabelEncoder()
X["hour"] = le.fit_transform(data["hour"])
X["nodeID"] = le.fit_transform(data["nodeID"])
X["node_type"] = le.fit_transform(data["node_type"])
X["has_leak"] = le.fit_transform(data["has_leak"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

# we have to use a REGRESSOR now, not classificator, since it is not a classification problem anymore because we
# have to predict continues values, not discrete values that can be divided in classes
# https://www.projectpro.io/article/classification-vs-regression-in-machine-learning/545
# https://www.springboard.com/blog/data-science/regression-vs-classification/

model = Pipeline([('scaler', StandardScaler()),
                      ('MLPR', MLPRegressor(random_state=1, max_iter=500))])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))
print(metrics.r2_score(y_test, y_pred))
#print(metrics.mean_squared_log_error(y_test, y_pred))
