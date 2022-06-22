from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import csv

def execute_classifier(input_filename, prefix_output_filename, confusion_matrix_name):
    print("Executing Decision Tree for "+prefix_output_filename+confusion_matrix_name+"... ")

    model = DecisionTreeClassifier()

    print("Loading " + input_filename + "...")

    data = pd.read_csv(input_filename,
                       names=["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
                              "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
                              "current_leak_demand_value","smart_sensor_presence"])

    print("Dividing X and y matrices...")

    X = data[["demand_value", "head_value", "pressure_value", "smart_sensor_presence"]].copy()
    y = data["has_leak"].astype(int)

    prediction_accuracy = 0.0

    conf_matrix_list_of_arrays = []
    kf = KFold(n_splits= 5, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        prediction_accuracy += accuracy_score(y_test, y_pred)

        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_list_of_arrays.append(conf_matrix)

    prediction_accuracy = prediction_accuracy / 5

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

    ax.set_title(confusion_matrix_name+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    filename = "decision_tree_results/"+prefix_output_filename+confusion_matrix_name+'.png'
    plt.savefig(filename)
    plt.clf()

    print("Decision Tree for " + prefix_output_filename + confusion_matrix_name + " done!\n")

    return prediction_accuracy, calc_accuracy, precision, recall, false_positive_rate

def run_evaluation():

    prefixes = ["1WEEK_","1MONTH_","1YEAR_"]

    folders = ["nodes_with_sensors_exported_month_large_complete_one_reservoirs_small/",
                "nodes_with_sensors_exported_month_large_complete_one_reservoirs_large/",
                "nodes_with_sensors_exported_month_large_complete_two_reservoirs/"]

    nodes_filename = "nodes_output.csv"

    for prefix in prefixes:

        output_filename = "decision_tree_results/" + prefix + 'decision_tree_accuracies.csv'

        # open the file in the write mode
        f = open(output_filename, 'w')

        # create the csv writer
        writer = csv.writer(f)

        header = ["VALUE","1 RES SMALL", "1 RES LARGE", "2 RES LARGE"]
        writer.writerow(header)

        input_f0 = folders[0] + prefix + nodes_filename
        input_f1 = folders[1] + prefix + nodes_filename
        input_f2 = folders[2] + prefix + nodes_filename

        f0_res = execute_classifier(input_f0, prefix, "decision_tree_1small")
        f1_res = execute_classifier(input_f1, prefix, "decision_tree_1large")
        f2_res = execute_classifier(input_f2, prefix, "decision_tree_2large")

        accuracy_row = ["ACCURACY",f0_res[1],f1_res[1],f2_res[1]]
        writer.writerow(accuracy_row)

        precision_row = ["PRECISION",f0_res[2],f1_res[2],f2_res[2]]
        writer.writerow(precision_row)

        # close the file
        f.close()



if __name__ == "__main__":
    print("Decision Tree comparison started...\n")

    run_evaluation()

    print("\nDecision Tree comparison done!\n")