import csv

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from joblib import dump, load

from pathlib import Path

import pandas as pd
import os

from itertools import combinations


def fit_model(X_train, y_train):
    print("Using StandardScaler to the data...")

    model = Pipeline([('scaler', StandardScaler()),
                      ('MLPR', MLPRegressor(random_state=1, max_iter=1500))])

    print("Fitting without k-fold...")

    model.fit(X_train, y_train)

    return model

def load_model(model_prefix, X_train, y_train, input_full_dataset, model_persistency=False):

    model_fitted = None

    if (model_persistency):
        print("\nModel persistency ENABLED")

        input_filename_full_fitted_model = ""
        output_filename_full_fitted_model = model_prefix + input_full_dataset.replace(".csv",
                                                                                      "") + '_full_fit_model.joblib'

        for filename in Path(".").glob("*.joblib"):
            input_filename_full_fitted_model = str(filename)

            # cheap hack. we just have one model file. break at first finding.
            break

        # output_filename_full_fitted_model = model_prefix + input_full_dataset.replace(".csv", "") + '_full_fit_model.joblib'

        if input_filename_full_fitted_model != "":
            print("Full fitted model already exists. Loading " + input_filename_full_fitted_model + "...")
            model_fitted = load(input_filename_full_fitted_model)
        else:
            # we first fit the model on the complete dataset and save the fitted model back
            print("No old model found. Fitting...")
            model_fitted = fit_model(X_train, y_train)

            dump(model_fitted, output_filename_full_fitted_model)
            print("Model saved to: " + output_filename_full_fitted_model)
    else:
        print("\nModel persistency DISABLED")
        model_fitted = fit_model(X_train, y_train)

    print("")

    return model_fitted

def fit_and_predict_on_full_dataset(folder_input, input_full_dataset, model_persistency, writer, model_prefix="", X_input=None):

    # hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,
    # has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand

    #TODO: write a csv for different features stats!!!

    complete_path = folder_input+input_full_dataset

    print("LOADING " + complete_path + "...")

    data = pd.read_csv(complete_path)

    print("Dividing X and y matrices...")

    X = None

    if(X_input is None):
        X = data.copy().drop(columns=["hour", "nodeID", "node_type", "has_leak", "demand_value"])

        le = preprocessing.LabelEncoder()
        X["hour"] = le.fit_transform(data["hour"])
        X["nodeID"] = le.fit_transform(data["nodeID"])
        X["node_type"] = le.fit_transform(data["node_type"])
        X["has_leak"] = le.fit_transform(data["has_leak"])

        # print("X_input is none")
    else:
        X = X_input

        # print("X_input is NOT none")

    y = data["demand_value"].copy()

    # print(list(X.columns))

    # MLP

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

    # we have to use a REGRESSOR now, not classificator, since it is not a classification problem anymore because we
    # have to predict continues values, not discrete values that can be divided in classes
    # https://www.projectpro.io/article/classification-vs-regression-in-machine-learning/545
    # https://www.springboard.com/blog/data-science/regression-vs-classification/

    model = load_model(model_prefix, X_train, y_train, input_full_dataset, model_persistency=model_persistency)

    # print(X.columns)

    print("Predicting...")

    y_pred = model.predict(X_test)

    print("\nPredict finished! Results:\n")

    # summary of the model
    print("Score: ", model.score(X_test, y_test))
    print("r2 score: ", metrics.r2_score(y_test, y_pred))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
    print()

    print("Executing k-fold...")
    scores = cross_val_score(model, X, y, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("")

def check_if_fresh_model_is_required(fresh_start):
    if (fresh_start):
        print("\nFresh start ENABLED. Deleting old models...\n")

        for filename in Path(".").glob("*.joblib"):
            try:
                os.remove(filename)
                print(str(filename) + " deleted")
            except OSError:
                print("\nError while deleting " + str(filename) + "\n")
        print("All models deleted.\n")
    else:
        print("\nFresh start NOT ENABLED. Will reuse old models if present.\n")

def run_with_different_inputs(folder_input, input_full_dataset):

    complete_path = folder_input + input_full_dataset

    output_filename = "multiple_features_mlp_results.csv"

    print("output_filename : ", output_filename)
    print("input dataset path : ", complete_path)

    # open the file in the write mode
    f = open(output_filename, "w", newline='', encoding='utf-8')

    # create the csv writer
    writer = csv.writer(f)

    header = ['base_demand', 'head_value', 'pressure_value', 'x_pos', 'y_pos',
              'leak_area_value', 'leak_discharge_value', 'current_leak_demand_value',
              'smart_sensor_is_present', 'tot_network_demand',
              'hour', 'nodeID', 'node_type', 'has_leak','r2_score','mser','dataset']

    writer.writerow(header)

    print("LOADING " + complete_path + "...")

    data = pd.read_csv(complete_path)

    le = preprocessing.LabelEncoder()
    data["hour"] = le.fit_transform(data["hour"])
    data["nodeID"] = le.fit_transform(data["nodeID"])
    data["node_type"] = le.fit_transform(data["node_type"])
    data["has_leak"] = le.fit_transform(data["has_leak"])

    # print("Dividing X and y matrices...")

    features = data.columns.copy()
    filter = ["demand_value"]
    features = features.drop(filter)

    for i in range(len(features)):
        oc = combinations(features, i + 1)
        for c in oc:
            # print(list(c), i, i + 1)
            # tmp.append(list(c))
            cols = list(c)

            X = data[cols].copy()

            # print(list(X.columns))

            model_persistency = True
            fresh_start = True

            folder_input = "datasets_for_mlp/"

            input_full_dataset = '1M_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'

            input_alt_dataset = '1M_ALT_one_res_small_with_leaks_rand_base_dem_nodes_output.csv'

            model_prefix = "MLP_model_"

            execute_analysis(model_persistency, fresh_start, folder_input, input_full_dataset, input_alt_dataset, model_prefix, writer, X_input=X)



    f.close()


def execute_analysis(model_persistency, fresh_start, folder_input, input_full_dataset, input_alt_dataset, model_prefix, writer, X_input = None):
    print("MLP Regression analysis started!")

    check_if_fresh_model_is_required(fresh_start)

    fit_and_predict_on_full_dataset(folder_input, input_full_dataset, model_persistency, writer, model_prefix=model_prefix, X_input=X_input)

    fit_and_predict_on_full_dataset(folder_input, input_alt_dataset, model_persistency, writer, model_prefix=model_prefix, X_input=X_input)

    print("\nMLP Regression analysis finished!")

if __name__ == "__main__":

    model_persistency = True
    fresh_start = True

    folder_input = "datasets_for_mlp/"

    input_full_dataset = '1M_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'

    input_alt_dataset = '1M_ALT_one_res_small_with_leaks_rand_base_dem_nodes_output.csv'

    model_prefix = "MLP_model_"

    execute_analysis(model_persistency, fresh_start, folder_input, input_full_dataset, input_alt_dataset, model_prefix)

    # run_with_different_inputs(folder_input, input_full_dataset)