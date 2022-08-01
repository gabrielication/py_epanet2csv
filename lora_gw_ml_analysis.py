from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from joblib import dump, load

import csv
import os
import pandas as pd
import numpy as np
import sys


def fit_on_complete_dataset(model, input_full_dataset, writer):
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

    print("Predicting on full dataset...")

    start_index_test = len(data_full["nodeID"].unique()) * 24 * 21

    X_test = X.iloc[start_index_test:]
    y_test = y.iloc[start_index_test:]

    y_pred = model.predict(X_test)

    produce_results_from_cf_matrix(y_test, y_pred, "full", writer)

    return model

def execute_classifier(model, input_gw_dataset, gw_id, writer):

    print("Loading " + input_gw_dataset + "...")

    data_gw = pd.read_csv(input_gw_dataset)
    # print(data_gw.head())
    """
     hour nodeID  demand_value  ...  current_leak_demand_value     gw_rssi  gw_sf
    """

    # data_gw = pd.read_csv("1M_one_res_large_nodes_output.csv",
    #                         names=["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
    #                                "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
    #                                "current_leak_demand_value", "smart_sensor_presence"])
    # print(data_gw.head())

    # cut dataset to 4 weeks
    end_index_dataset = len(data_gw["nodeID"].unique()) * 24 * 28
    start_index_test = len(data_gw["nodeID"].unique()) * 24 * 21
    data_gw = data_gw.iloc[start_index_test:end_index_dataset]
    data_gw.reset_index(inplace=True, drop=True)

    print("Cuting " + input_gw_dataset + "...")
    # print(data_gw.head())

    print("Dividing X_gw and y_gw matrices...")
    X_test_gw = data_gw[["demand_value", "head_value", "pressure_value"]].copy()
    y_test_gw = data_gw["has_leak"].astype(int)

    # test dataset represent only the last week
    print("Predicting...")

    y_pred_test_gw = model.predict(X_test_gw)

    data_gw['y_pred_test_gw'] = y_pred_test_gw


    print("len all : ", len(y_pred_test_gw), " - ", len(X_test_gw))
    print("len all dataset : ", len(data_gw))

    idx  = data_gw.index[data_gw['gw_sf'] == 7].tolist()

    print("len only covered node : ", len(idx))

    y_test_gw = y_test_gw.iloc[idx]
    y_pred_test_gw = y_pred_test_gw[idx]

    print("len all 2 : ", len(y_pred_test_gw), " - ", len(y_test_gw))


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

    print(calc_accuracy,precision,recall,fscore,support,false_positive_rate)

    out_row = [gw_id, calc_accuracy, precision, recall, fscore, false_positive_rate, true_negatives, false_positives, false_negatives, true_positives]
    writer.writerow(out_row)


def execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix="", model_out_prefix=""):
    output_filename = model_out_prefix+'lora_all_gw_ml_results_'+input_full_dataset


    print("output_filename : ",output_filename)
    print("input_full_dataset : ", input_full_dataset)

    # open the file in the write mode
    f = open(output_filename, "w", newline='', encoding='utf-8')

    # create the csv writer
    writer = csv.writer(f)

    header = ["gw_id","accuracy","precision","recall","fscore","false_positive_rate","true_negatives","false_positives","false_negatives","true_positives"]
    writer.writerow(header)

    output_filename_full_fitted_model = model_out_prefix + input_full_dataset.replace(".csv","") + '_full_fit_model.joblib'

    print(output_filename_full_fitted_model)

    if os.path.exists(output_filename_full_fitted_model):
        print("\nFull fitted model already exists. Loading "+output_filename_full_fitted_model+"...")
        model_fitted = load(output_filename_full_fitted_model)
    else:
        # we first fit the model on the complete dataset and save the fitted model back
        model_fitted = fit_on_complete_dataset(model, input_full_dataset, writer)
        dump(model_fitted, output_filename_full_fitted_model)

    # we have to iterate each gateway and produce a report from its prediction
    for gw_id in range(10, 49):
        index = str(gw_id)
        input_gw_dataset = folder_prefix + "gw_" + index + "_lora" + input_full_dataset
        print("\n\n *******input_gw_dataset : ", input_gw_dataset)
        print(gw_id)
        if os.path.exists(input_gw_dataset):
            execute_classifier(model_fitted, input_gw_dataset, gw_id, writer)

    f.close()

if __name__ == "__main__":
    print("Lora GW ML benchmark started...\n")

    # DECISION TREE

    input_full_dataset = "1M_one_res_small_nodes_output.csv"
    folder_prefix = "lora_gw_datasets/"

    model = Pipeline([('scaler', StandardScaler()), ('DTC', DecisionTreeClassifier())])
    execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "dt_")

    # input_full_dataset = "1M_one_res_large_nodes_output.csv"
    # folder_prefix = "lora_gw_datasets/"
    #
    # model = Pipeline([('scaler', StandardScaler()), ('DTC', DecisionTreeClassifier())])
    # execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "dt_")

    # input_full_dataset = "1M_two_res_large_nodes_output.csv"
    # folder_prefix = "lora_gw_datasets/"
    #
    # model = Pipeline([('scaler', StandardScaler()), ('DTC', DecisionTreeClassifier())])
    # execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "dt_")
    #
    # # MLP
    #
    # input_full_dataset = "1M_one_res_small_nodes_output.csv"
    # folder_prefix = "lora_gw_datasets/"
    #
    # model = Pipeline([('scaler', StandardScaler()),
    #                   ('MLPC', MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(100)))])
    # execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "mlp_1l_100u_")
    #
    # model = Pipeline([('scaler', StandardScaler()),
    #                   ('MLPC', MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(100, 100)))])
    # execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "mlp_2l_100u_")
    #
    #
    #input_full_dataset = "1M_one_res_large_nodes_output.csv"
    #folder_prefix = "lora_gw_datasets/"
    #
    #model = Pipeline([('scaler', StandardScaler()),
    #                  ('MLPC', MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(2000)))])
    #execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "mlp_1l_2000u_")
    #
    # model = Pipeline([('scaler', StandardScaler()),
    #                   ('MLPC', MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(2000, 2000)))])
    # execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "mlp_2l_2000u_")
    #
    #
    # input_full_dataset = "1M_two_res_large_nodes_output.csv"
    # folder_prefix = "lora_gw_datasets/"
    #
    # model = Pipeline([('scaler', StandardScaler()),
    #                   ('MLPC', MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(4000)))])
    # execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "mlp_1l_4000u_")
    #
    # model = Pipeline([('scaler', StandardScaler()),
    #                   ('MLPC', MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(4000, 4000)))])
    # execute_classifier_for_each_gw(model, input_full_dataset, folder_prefix, "mlp_2l_4000u_")


    print("\nLora GW ML benchmark done!\n")