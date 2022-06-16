from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    SVC(kernel="rbf",gamma='auto'),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def execute_classifier(model, name, X, y):
    print("Executing "+name+"...")

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

    #print(mean_of_conf_matrix_arrays)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    mean_of_conf_matrix_arrays.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         mean_of_conf_matrix_arrays.flatten() / np.sum(mean_of_conf_matrix_arrays)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(mean_of_conf_matrix_arrays, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    filename = name+'.png'
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    print("Plot classifier comparison started!\n")
    filename = 'sample_nodes_output_one_week.csv'
    print("Loading csv...")

    data = pd.read_csv(filename, names=["hour","nodeID","demand_value","head_value","pressure_value","x_pos", "y_pos",
                                        "node_type", "has_leak", "leak_area_value", "leak_discharge_value", "current_leak_demand_value"])

    print("Dividing X and y matrices...\n")

    X = data[["demand_value","head_value","pressure_value"]].copy()
    y = data["has_leak"].astype(int)

    for name, clf in zip(names, classifiers):
        execute_classifier(clf,name,X,y)

    print("\nComparison done!")