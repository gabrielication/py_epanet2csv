#WARNING: sklearnex 2021.5.3 works only with sklearn 1.0.2
'''
from sklearnex import patch_sklearn
patch_sklearn()
'''

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filename = 'P1_nodes_output.csv'

print("Loading csv...")

data = pd.read_csv(filename, names=["hour","nodeID","demand_value","head_value","pressure_value","x_pos", "y_pos",
                                    "node_type", "has_leak", "leak_area_value", "leak_discharge_value", "current_leak_demand_value"])

print("Dividing X and y matrices...")

X = data[["demand_value","head_value","pressure_value"]].copy()
y = data["has_leak"].astype(int)

model = KNeighborsClassifier()
#model = svm.SVC(kernel="rbf", gamma=0.7, C=1.0)
#model = svm.SVC(kernel="linear")

conf_matrix_list_of_arrays = []
kf = KFold(n_splits=5, random_state=1, shuffle=True)
for train_index, test_index in kf.split(X):

   X_train, X_test = X.iloc[train_index], X.iloc[test_index]
   y_train, y_test = y.iloc[train_index], y.iloc[test_index]

   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

   conf_matrix = confusion_matrix(y_test, y_pred)
   conf_matrix_list_of_arrays.append(conf_matrix)

mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)

print(mean_of_conf_matrix_arrays)

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in
                mean_of_conf_matrix_arrays.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     mean_of_conf_matrix_arrays.flatten()/np.sum(mean_of_conf_matrix_arrays)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

ax = sns.heatmap(mean_of_conf_matrix_arrays, annot=labels, fmt='', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.savefig('prova.png')
plt.show()

'''
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print('After SMOTE, the shape of train_X: {}'.format(X_res.shape))
print('After SMOTE, the shape of train_y: {} \n'.format(y_res.shape))

print("After SMOTE, counts of label '1': {}".format(sum(y_res == 1)))
print("After SMOTE, counts of label '0': {}".format(sum(y_res == 0)))

model = svm.SVC(kernel="linear")

model.fit(X_res,y_res)

target_y_pred = model.predict(X_test)

accuracy_res=np.round(accuracy_score(y_test,target_y_pred),2)
precision_res= np.round(precision_score(y_test,target_y_pred),2)
recall_res= np.round(recall_score(y_test,target_y_pred),2)

print(accuracy_res,precision_res,recall_res)
'''

'''
print(np.unique(y_train))
# Create linear regression object
#regr = svm.SVC(kernel="linear")
regr = svm.SVC(kernel="rbf", gamma=0.7, C=1.0)

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