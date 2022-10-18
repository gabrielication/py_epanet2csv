import os

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing

from joblib import dump, load

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        print("Model persistency ENABLED")
        output_filename_full_fitted_model = model_prefix + input_full_dataset.replace(".csv", "") + '_full_fit_model.joblib'

        if os.path.exists(output_filename_full_fitted_model):
            print("Full fitted model already exists. Loading " + output_filename_full_fitted_model + "...")
            model_fitted = load(output_filename_full_fitted_model)
        else:
            # we first fit the model on the complete dataset and save the fitted model back
            model_fitted = fit_model(X_train, y_train)

            dump(model_fitted, output_filename_full_fitted_model)
            print("Model saved to: " + output_filename_full_fitted_model)
    else:
        print("Model persistency DISABLED")
        model_fitted = fit_model(X_train, y_train)

    return model_fitted

def fit_and_predict_on_full_dataset(input_full_dataset):
    print("Loading csv...")

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

    model = load_model(model_prefix, X_train, y_train, input_full_dataset, model_persistency=True)

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

if __name__ == "__main__":
    print("Lora GW ML benchmark started...\n")

    model_persistency = False
    model_prefix = "MLP_model_"

    input_full_dataset = '1M_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'

    fit_and_predict_on_full_dataset(input_full_dataset)

