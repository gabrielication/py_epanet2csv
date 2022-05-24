#WARNING: sklearnex 2021.5.3 works only with sklearn 1.0.2
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import csv

names = [
    "KNeighborsClassifier",
    "LinearSVM",
    "RBFSVM",
    "GaussianProcess",
    "DecisionTree",
    "RandomForest",
    "MLPClassifier",
    "AdaBoostClassifier",
    "GaussianNaiveBayes",
    "QuadraticDiscriminantAnalysis",
]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

filename = 'sample_nodes_output_one_week.csv'

print("Loading csv...")

data = pd.read_csv(filename, names=["hour","nodeID","demand_value","head_value","pressure_value","x_pos", "y_pos",
                                    "node_type", "has_leak", "leak_area_value", "leak_discharge_value", "current_leak_demand_value"])

print("Dividing X and y matrices...")

X = data[["demand_value","head_value","pressure_value"]].copy()
X = X.values
y = data["has_leak"].astype(int)

max_classifier = ""
max_accuracy = 0.0
max_std = 0.0

outName = "classifiers_results.csv"
out = open(outName, "w")
writer = csv.writer(out)

# iterate over classifiers
for name, clf in zip(names, classifiers):
    score = cross_val_score(clf, X, y, scoring='accuracy', cv=10)

    accuracy_score = score.mean()
    std_score = score.std()

    output_row = [name,accuracy_score,std_score]
    writer.writerow(output_row)

    if(accuracy_score > max_accuracy):
        max_accuracy = accuracy_score
        max_std = std_score
        max_classifier = name

    print(name,accuracy_score,std_score)

out.close()
print("\nBest classifier is: "+ max_classifier+"\nMax accuracy is: "+str(max_accuracy)+"\nMax std is: "+str(max_std))