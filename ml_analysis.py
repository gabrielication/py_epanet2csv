#WARNING: sklearnex 2021.5.3 works only with sklearn 1.0.2
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

filename = 'sample_nodes_output_one_week.csv'

print("Loading csv...")

data = pd.read_csv(filename, names=["hour","nodeID","demand_value","head_value","pressure_value","x_pos", "y_pos",
                                    "node_type", "has_leak", "leak_area_value", "leak_discharge_value", "current_leak_demand_value"])
'''
target_y = data[["has_leak","leak_area_value","leak_discharge_value"]]
attributes_X = data[["demand_value","head_value","pressure_value","current_leak_demand_value"]]
'''

print("Dividing X and y matrices...")

X = data[["demand_value","head_value","pressure_value"]].copy()
y = data["has_leak"].astype(int)

#target_y = data[["has_leak","leak_area_value","leak_discharge_value","current_leak_demand_value"]].copy()
#attributes_X["tot_demand"] = data["demand_value"] + data["current_leak_demand_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print(np.unique(y_train))
# Create linear regression object
regr = svm.SVC(kernel="linear")

print("Training started...")
# Train the model using the training sets
regr.fit(X_train, y_train)

print("Score on X_train and y_train is: "+str(regr.score(X_train, y_train)))
print("Score on X_test and y_test is: "+str(regr.score(X_test, y_test)))

print("Prediction started...")

# Make predictions using the testing set
target_y_pred1 = regr.predict(X_train)
target_y_pred2 = regr.predict(X_test)

print("Accuracy score is: "+str(accuracy_score(y_test, target_y_pred2)))

'''
print("Prediction started...")
# Make predictions using the testing set
target_y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, target_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, target_y_pred))
'''
#print("debug")