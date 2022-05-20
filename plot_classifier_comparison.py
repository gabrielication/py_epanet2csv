#WARNING: sklearnex 2021.5.3 works only with sklearn 1.0.2
from sklearnex import patch_sklearn
patch_sklearn()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

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

figure = plt.figure(figsize=(27, 9))

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name+" classifier's score: "+str(score))