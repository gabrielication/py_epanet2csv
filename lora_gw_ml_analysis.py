from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import csv
import os
import pandas as pd
import numpy as np


def fit_on_complete_dataset(input_full_dataset):
    model = Pipeline([('scaler', StandardScaler()), ('DTC', DecisionTreeClassifier())])

    print("Loading " + input_full_dataset + "...")

    data_full = pd.read_csv(input_full_dataset,
                            names=["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
                                   "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
                                   "current_leak_demand_value", "smart_sensor_presence"])

    print("Dividing X and y matrices...")

    # cut dataset to 4 weeks
    end_index_dataset = len(data_full["nodeID"].unique()) * 24 * 28
    data_full = data_full.iloc[:end_index_dataset]

    X = data_full[["demand_value", "head_value", "pressure_value"]].copy()
    y = data_full["has_leak"].astype(int)

    # train dataset represents only the first three weeks
    end_index_train = len(data_full["nodeID"].unique()) * 24 * 21

    X_train = X.iloc[:end_index_train]

    y_train = y.iloc[:end_index_train]

    print("Fitting on the whole dataset...")

    model.fit(X_train, y_train)

    return model

def execute_classifier(model, input_gw_dataset, gw_id, writer):

    print("Loading " + input_gw_dataset + "...")

    data_gw = pd.read_csv(input_gw_dataset)

    # cut dataset to 4 weeks
    end_index_dataset = len(data_gw["nodeID"].unique()) * 24 * 28

    data_gw = data_gw.iloc[:end_index_dataset]

    print("Dividing X_gw and y_gw matrices...")

    X_gw = data_gw[["demand_value", "head_value", "pressure_value"]].copy()
    y_gw = data_gw["has_leak"].astype(int)

    # test dataset represent only the last week

    start_index_test = len(data_gw["nodeID"].unique()) * 24 * 21

    X_test_gw = X_gw.iloc[start_index_test:]
    y_test_gw = y_gw.iloc[start_index_test:]

    print("Predicting...")

    y_pred_test_gw = model.predict(X_test_gw)

    produce_results_from_cf_matrix(y_test_gw, y_pred_test_gw, gw_id, writer)

def produce_results_from_cf_matrix(y_test, y_pred, gw_id, writer):
    cf_matrix = confusion_matrix(y_test, y_pred)

    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]

    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    true_positives = 0.0

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')

    try:
        #prediction might output just true negatives if all values are 0. we catch the exception here

        true_negatives = float(group_counts[0])
        false_positives = float(group_counts[1])
        false_negatives = float(group_counts[2])
        true_positives = float(group_counts[3])
    except:
        print("All true negatives")

    calc_accuracy = (true_positives + true_negatives) / (
            true_positives + true_negatives + false_negatives + false_positives)

    false_positive_rate = false_positives / (false_positives + true_negatives)

    #print(calc_accuracy,precision,recall,fscore,support,false_positive_rate)

    out_row = [gw_id, calc_accuracy, precision, recall, fscore, false_positive_rate, true_negatives, false_positives, false_negatives, true_positives]
    writer.writerow(out_row)


def execute_classifier_for_each_gw(input_full_dataset, folder_prefix=""):
    output_filename = 'lora_all_gw_ml_results_'+input_full_dataset

    # open the file in the write mode
    f = open(output_filename, 'w+')

    # create the csv writer
    writer = csv.writer(f)

    header = ["gw_id","accuracy","precision","recall","fscore","false_positive_rate","true_negatives","false_positives","false_negatives","true_positives"]
    writer.writerow(header)

    # we first fit the model on the complete dataset and save the fitted model back
    model = fit_on_complete_dataset(input_full_dataset)

    # we have to iterate each gateway and produce a report from its prediction
    gw_id = 0
    index = str(gw_id)
    input_gw_dataset = folder_prefix+"gw_" + index + "_lora" + input_full_dataset

    while os.path.exists(input_gw_dataset):
        execute_classifier(model, input_gw_dataset, gw_id, writer)

        gw_id += 1
        index = str(gw_id)
        input_gw_dataset = folder_prefix+"gw_" + index + "_lora" + input_full_dataset

    f.close()

if __name__ == "__main__":
    print("Decision Tree benchmark started...\n")

    execute_classifier_for_each_gw("1M_two_res_large_nodes_output.csv", "lora_gw_datasets/")

    print("\nDecision Tree benchmark done!\n")