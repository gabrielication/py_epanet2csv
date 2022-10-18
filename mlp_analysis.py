from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing

from joblib import dump, load

from pathlib import Path

import pandas as pd
import os

def fit_model(X_train, y_train):
    print("Using StandardScaler to the data...")

    model = Pipeline([('scaler', StandardScaler()),
                      ('MLPR', MLPRegressor(random_state=1, max_iter=500))])

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
            print(input_filename_full_fitted_model)
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

def fit_and_predict_on_full_dataset(input_full_dataset, model_persistency):
    print("LOADING "+input_full_dataset+"...")

    # hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,
    # has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand

    data = pd.read_csv(input_full_dataset)

    print("Dividing X and y matrices...")

    X = data.copy().drop(columns=["hour", "nodeID", "node_type", "has_leak", "demand_value"])
    y = data["demand_value"].copy()

    # MLP

    le = preprocessing.LabelEncoder()
    X["hour"] = le.fit_transform(data["hour"])
    X["nodeID"] = le.fit_transform(data["nodeID"])
    X["node_type"] = le.fit_transform(data["node_type"])
    X["has_leak"] = le.fit_transform(data["has_leak"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

    # we have to use a REGRESSOR now, not classificator, since it is not a classification problem anymore because we
    # have to predict continues values, not discrete values that can be divided in classes
    # https://www.projectpro.io/article/classification-vs-regression-in-machine-learning/545
    # https://www.springboard.com/blog/data-science/regression-vs-classification/

    model = load_model(model_prefix, X_train, y_train, input_full_dataset, model_persistency=model_persistency)

    print("Predicting...")

    y_pred = model.predict(X_test)

    print("\nPredict finished! Results:\n")

    # summary of the model
    print("Score: ", model.score(X_test, y_test))
    print("r2 score: ", metrics.r2_score(y_test, y_pred))
    print()

    print("Executing k-fold...")
    scores = cross_val_score(model, X, y, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("")

def run_fresh(fresh_start):
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

if __name__ == "__main__":
    print("MLP Regression analysis started!")

    model_persistency = True
    fresh_start = True

    model_prefix = "MLP_model_"

    run_fresh(fresh_start)

    input_full_dataset = '1d_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'

    fit_and_predict_on_full_dataset(input_full_dataset, model_persistency)

    input_alt_dataset = '1d_alt_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'

    fit_and_predict_on_full_dataset(input_alt_dataset, model_persistency)

    print("\nMLP Regression analysis finished!")

