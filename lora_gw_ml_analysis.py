from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_and_show_cf_matrix(y_test, y_pred_test):
    cf_matrix = confusion_matrix(y_test, y_pred_test)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    true_negatives = float(group_counts[0])
    false_positives = float(group_counts[1])
    false_negatives = float(group_counts[2])
    true_positives = float(group_counts[3])

    calc_accuracy = (true_positives + true_negatives) / (
            true_positives + true_negatives + false_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)

    print(calc_accuracy, precision, recall, false_positive_rate)

    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


def execute_classifier(input_full_dataset, input_gw_dataset):

    model = Pipeline([('scaler', StandardScaler()), ('DTC', DecisionTreeClassifier())])

    print("Loading " + input_full_dataset + "...")

    data_full = pd.read_csv(input_full_dataset,
                       names=["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
                              "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
                              "current_leak_demand_value", "smart_sensor_presence"])

    print("Dividing X and y matrices...")

    X = data_full[["demand_value", "head_value", "pressure_value"]].copy()
    y = data_full["has_leak"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("Fitting on the whole dataset...")

    model.fit(X_train, y_train)

    # Make predictions using the testing set

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("Loading " + input_gw_dataset + "...")

    data_gw = pd.read_csv(input_gw_dataset)

    print("Dividing X_gw and y_gw matrices...")

    X_gw = data_gw[["demand_value", "head_value", "pressure_value"]].copy()
    y_gw = data_gw["has_leak"].astype(int)

    X_train_gw, X_test_gw, y_train_gw, y_test_gw = train_test_split(X_gw, y_gw, test_size=0.3)

    print("Predicting...")

    y_pred_test_gw = model.predict(X_test_gw)

    make_and_show_cf_matrix(y_test_gw,y_pred_test_gw)

if __name__ == "__main__":
    print("Decision Tree benchmark started...\n")

    execute_classifier("1M_two_res_large_nodes_output.csv","gw_0_lora1M_two_res_large_nodes_output.csv")

    print("\nDecision Tree benchmark done!\n")