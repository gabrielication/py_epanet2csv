
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
from sklearn.model_selection import train_test_split

filename = 'nodes_output.csv'

data = pd.read_csv(filename, names=["hour","nodeID","demand_value","head_value","pressure_value","x_pos", "y_pos",
                                    "node_type", "has_leak", "leak_area_value", "leak_discharge_value", "current_leak_demand_value"])

#TODO: add more fields to target_ylo
#TODO: when a leak is present, in the training set we should have demand replaced with current_leak_demand

target_y = data.drop(["hour","nodeID","demand_value","head_value","pressure_value","x_pos", "y_pos",
                                    "node_type","current_leak_demand_value"],axis=1)
attributes_X = data.drop(["hour","nodeID","has_leak", "x_pos", "y_pos","leak_area_value", "leak_discharge_value","node_type"],axis=1)

#data.info()
attributes_X.info()
target_y.info()

X_train, X_test, y_train, y_test = train_test_split(attributes_X, target_y,test_size=0.2)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
target_y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, target_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, target_y_pred))