from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

from mlxtend.classifier import EnsembleVoteClassifier

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

names = [
    "KNeighborsClassifier",
    "LinearSVM",
    "RBFSVM",
    "DecisionTree",
    "RandomForest",
    "AdaBoostClassifier",
    "GaussianNaiveBayes",
    "Ensemble",
]

def execute_classifier(model, name, k_folds, X, y, prefix_output_filename):
    print("Executing "+name+"...")

    prediction_accuracy = 0.0

    conf_matrix_list_of_arrays = []
    kf = KFold(n_splits= k_folds, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        prediction_accuracy += accuracy_score(y_test, y_pred)

        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_list_of_arrays.append(conf_matrix)

    prediction_accuracy = prediction_accuracy / k_folds

    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)

    #print(mean_of_conf_matrix_arrays)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    mean_of_conf_matrix_arrays.flatten()]

    true_negatives = float(group_counts[0])
    false_positives = float(group_counts[1])
    false_negatives = float(group_counts[2])
    true_positives = float(group_counts[3])

    calc_accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)

    group_percentages = ["{0:.2%}".format(value) for value in
                         mean_of_conf_matrix_arrays.flatten() / np.sum(mean_of_conf_matrix_arrays)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(mean_of_conf_matrix_arrays, annot=labels, fmt='', cmap='Blues')

    ax.set_title(name+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    filename = prefix_output_filename+name+'.png'
    plt.savefig(filename)
    plt.clf()
    return prediction_accuracy, calc_accuracy, precision, recall, false_positive_rate

def execute_classifier_comparison(input_filename, prefix_output_filename):
    print("Loading "+input_filename+"...")

    data = pd.read_csv(input_filename,
                       names=["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
                              "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
                              "current_leak_demand_value"])

    output_filename = prefix_output_filename+'prediction_accuracies.csv'

    # open the file in the write mode
    f = open(output_filename, 'w')

    # create the csv writer
    writer = csv.writer(f)

    header = ["Classificator","Accuracy","Calc Accuracy", "Precision", "Recall", "False Positive Rate"]
    writer.writerow(header)

    print("Dividing X and y matrices...\n")

    X = data[["demand_value", "head_value", "pressure_value"]].copy()
    y = data["has_leak"].astype(int)

    clf1 = KNeighborsClassifier()
    clf2 = SVC(kernel="linear")
    clf3 = SVC(kernel="rbf",gamma='auto')
    clf4 = DecisionTreeClassifier()
    clf5 = RandomForestClassifier()
    clf6 = AdaBoostClassifier()
    clf7 = GaussianNB()

    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4, clf5, clf6, clf7], weights=[1, 1, 1, 1, 1, 1, 1], voting="hard")

    classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf]

    for name, clf in zip(names, classifiers):
        prediction_measurements = execute_classifier(clf, name, 5, X, y, prefix_output_filename)
        #print(name + "'s prediction accuracy (mean of kfolds) is: " + str(prediction_accuracy) + "\n")

        output_row = [name] + list(prediction_measurements)
        print(output_row)

        writer.writerow(output_row)

    # close the file
    f.close()

    print("\n"+input_filename+" comparison done!\n\n")

if __name__ == "__main__":
    print("Plot classifier comparison started!\n")

    #input_filename = 'nodes_output.csv'

    execute_classifier_comparison("exported_month_large_complete_one_reservoirs_small/1WEEK_nodes_output.csv", "exported_month_large_complete_one_reservoirs_small/1WEEK_")
    execute_classifier_comparison("exported_month_large_complete_one_reservoirs_small/1MONTH_nodes_output.csv", "exported_month_large_complete_one_reservoirs_small/1MONTH_")
    execute_classifier_comparison("exported_month_large_complete_one_reservoirs_small/1YEAR_nodes_output.csv", "exported_month_large_complete_one_reservoirs_small/1YEAR_")

