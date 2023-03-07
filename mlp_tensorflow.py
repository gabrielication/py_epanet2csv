import numpy as np
import pandas as pd
import os
import shutil
import csv
import seaborn as sns
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers

from keras_visualizer import visualizer

def is_gpu_supported():
    gpu_list = tf.config.list_physical_devices('GPU')

    if (len(gpu_list) == 0):
        print("GPU IS NOT SUPPORTED/ACTIVE/DETECTED!")

        return False
    else:
        print("GPU SUPPORTED: ", gpu_list)

        return True

def clean_old_models(model_path):
    print("FRESH START ENABLED. Cleaning ALL old models and their files...")

    # for filename in Path(".").glob("*.png"):
    #     try:
    #         os.remove(filename)
    #         print(str(filename) + " deleted")
    #     except OSError:
    #         print("\nError while deleting " + str(filename) + "\n")

    for filename in Path(".").glob(model_path):
        try:
            shutil.rmtree(filename)

            print(str(filename) + " deleted")
        except OSError:
            print("\nError while deleting " + str(filename) + "\n")

    print("All old files deleted.\n")

def count_nodes_from_dataframe(df):
    n_nodes = df['hour'].value_counts()['0:00:00']
    return n_nodes

def count_sim_duration_from_dataframe(df):
    duration = int(df['hour'].iloc[-1].split(":")[0])
    return duration

def load_model(model_path_filename, history_path_filename):

    if os.path.exists(model_path_filename):
        print("Model already exists!\nIf tensorflow versions from saved one differ then a crash might happen!")
        model = tf.keras.models.load_model(model_path_filename)

        history_complete_path = model_path_filename+'/'+history_path_filename+'.npy'

        history = np.load(history_complete_path, allow_pickle='TRUE').item()

        print("Loading model: "+model_path_filename+"/\nLoading history: "+history_complete_path)

        return model, history
    else:
        print("Previous fitted model not found!")

        return None, None

def save_model(model, history, model_path_filename, history_path_filename):
    model.save(model_path_filename)

    history_complete_path = model_path_filename+'/'+history_path_filename+'.npy'

    np.save(history_complete_path, history.history)

    print("Model saved to: " + model_path_filename)
    print("Training History saved to: " + history_complete_path)

def formatted_datetime():
    # current date and time
    now = str(datetime.now())
    now = now.replace(" ", "_")
    now = now.replace(".", "_")
    now = now.replace(":", "_")

    return now

def load_dataset(complete_path, features, labels, slice_data=0.8):
    print("LOADING " + complete_path + "...")

    # We read our entire dataset
    data = pd.read_csv(complete_path)

    temp_cols = features.copy()
    temp_cols.append(labels)

    # We drop these columns because they are strings, booleans and other incompatible types. We will convert them later

    print("Extracting only columns: ", temp_cols)

    data_trans = data[temp_cols].copy()
    data_scaled = data_trans

    print("Dividing FEATURES (", features, ") and LABELS (", labels, ")...")

    # This was used in Tensorflow wiki but it's not the same as train test split. It will pick a SAMPLE jumping rows, not a clean SPLIT
    # train_dataset = data_scaled.sample(frac=0.8, random_state=0)

    duration = count_sim_duration_from_dataframe(data)
    n_nodes = count_nodes_from_dataframe(data)

    duration_percentage = int(slice_data * duration)

    train_dataset_size = duration_percentage * n_nodes

    train_dataset = data_scaled.iloc[:train_dataset_size, :]
    test_dataset = data_scaled.drop(train_dataset.index)

    # Tensorflow guide (https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb#scrollTo=2l7zFL_XWIRu)
    # says that the features are the columns that we want our network to train and labels is the value(s) to predict
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    # These instructions modificate also original dataframes
    train_labels = train_features.pop(labels)
    test_labels = test_features.pop(labels)

    return train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, duration, n_nodes

def process_dataset(folder_path, dataset_filename, output_filename):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    data = pd.read_csv(folder_path+dataset_filename)

    sim_duration = count_sim_duration_from_dataframe(data)
    n_nodes = count_nodes_from_dataframe(data)

    filtered_data = data.groupby('hour').head(10)
    base_demands = filtered_data.groupby("hour")["base_demand"].apply(list)
    demand_values = filtered_data.groupby("hour")["demand_value"].apply(list)
    head_values = filtered_data.groupby("hour")["head_value"].apply(list)
    pressure_values = filtered_data.groupby("hour")["pressure_value"].apply(list)
    has_leaks = filtered_data.groupby("hour")["has_leak"].apply(list)

    df = pd.DataFrame(columns=['list_of_bd', 'list_of_dv', 'list_of_hd', 'list_of_pr', 'has_leak'])

    for i in range(0, sim_duration+1):
        timestamp = str(i) + ":00:00"
        row = [base_demands.get(timestamp), demand_values.get(timestamp), head_values.get(timestamp),
               pressure_values.get(timestamp), True in has_leaks.get(timestamp)]

        # add the row to the DataFrame using .append()
        df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

    # complete_out_filename = output_filename
    #
    # df.to_csv(complete_out_filename, index=False)

    # print("Processed dataset saved to: "+complete_out_filename)
    #
    # return complete_out_filename, df

    n_rows = df.shape[0]
    cut_dataset_at = int(n_rows/2)

    first_half_dataset = df.iloc[:cut_dataset_at, :]
    snd_half_dataset = df.drop(first_half_dataset.index)

    return first_half_dataset, snd_half_dataset



def create_classifier_nn_model(train_features, n_nodes):
    print("Building Classifier Neural Network Model...")
    print("NORMALIZATION IS ENABLED!")
    # We want to Normalize (scale) the data since it can be too different in ranges
    # These lines will create a NORMALIZATION layer

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    input_layer = normalizer

    fst_level = n_nodes * 5
    snd_level = n_nodes * 3
    trd_level = n_nodes

    # Let's build the model. The first layer will be the normalizer that we built before
    # Depth=3 and Width=fst,snd,trd

    # The sigmoid function maps any input value to a range between 0 and 1, which can be interpreted
    # as the probability of the positive class. This is appropriate for binary classification problems.
    model = keras.Sequential([
        input_layer,
        # layers.Dense(fst_level, activation='relu', input_dim=train_features.shape[1]),
        layers.Dense(fst_level, activation='relu'),
        layers.Dense(snd_level, activation='relu'),
        layers.Dense(trd_level, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # binary_crossentropy loss function is commonly
    # used for binary classification problems because it is a
    # measure of the dissimilarity between the predicted probability
    # distribution and the true binary labels.

    lossfn = 'binary_crossentropy'

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt = tf.keras.optimizers.Adam()

    # These are the metrics for a binary classification problem
    metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=1)]

    model.compile(loss=lossfn,
                  metrics=metrics,
                  optimizer=opt)

    # Can be skipped
    model.summary()

    return model


def create_regression_nn_model(train_features, n_nodes):
    print("Building REGRESSION Neural Network Model...")

    if (len(train_features.columns) == 1):
        col = train_features.columns[0]
        bdem = np.array(train_features[col])

        normalizer = layers.Normalization(input_shape=[1, ], axis=None)
        normalizer.adapt(bdem)
    else:
        normalizer = tf.keras.layers.Normalization(axis=-1)
        #
        normalizer.adapt(np.array(train_features))
        #
        normalizer.mean.numpy()

    input_layer = normalizer

    fst_level = n_nodes * 5
    snd_level = n_nodes * 3
    trd_level = n_nodes

    # Let's build the model. The first layer will be the normalizer that we built before
    # Depth=3 and Width=fst,snd,trd
    # Since it is a Regression problem, last one will be a Linear output (funzione immagine)
    model = keras.Sequential([
        input_layer,
        # layers.Dense(fst_level, activation='relu', input_dim=train_features.shape[1]),
        layers.Dense(fst_level, activation='relu'),
        layers.Dense(snd_level, activation='relu'),
        layers.Dense(trd_level, activation='relu'),
        layers.Dense(1)
    ])

    lossfn = 'mean_squared_error'

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt = tf.keras.optimizers.Adam()

    metrics = ['mse', 'mae', tfa.metrics.r_square.RSquare()]

    # metrics = []

    # In regression we use mserr as loss funct
    # Adam is a good optimizer since it just do the calculation for SGD's parameters automatically starting from a fixed value
    model.compile(loss=lossfn,
                  metrics=metrics,
                  optimizer=opt)

    # Can be skipped
    model.summary()

    return model


def perform_neural_network_fit(model, train_features, train_labels, epochs, batch_size=None, validation_split=0.0,
                               callbacks=[None], verbose=1):
    # This array saves all the values obtained through the epochs

    print("epochs: ", epochs, "batch_size: ", batch_size, "validation_split: ", validation_split)

    print("Fitting...")

    history = model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
        callbacks=callbacks
    )

    print("Fitting finished.")

    return history


def evaluate_regression_nn_after_fit(model, test_features, test_labels):
    print("Evaluation started...")

    evl = model.evaluate(test_features, test_labels, verbose=0)

    loss = evl[0]
    mse = evl[1]
    mae = evl[2]
    r_square = evl[3]

    print("loss: ", loss)
    print("mse: ", mse)
    print("mae: ", mae)
    print("r_square: ", r_square)

    return loss, mse, mae, r_square


def predict_and_collect_results(model, test_features):
    print("Prediction started...")

    test_predictions = model.predict(test_features).flatten()

    return test_predictions

def create_or_load_nn_regressor(folder_path, dataset_filename, epochs, features, labels, batch_size=None,
                                model_path_filename="", history_path_filename="", slice_data=0.8,
                                validation_split=0.2, save_model_bool=False, fresh_start=False, evaluate_model=True):

    if(fresh_start):
        clean_old_models(model_path_filename)

    complete_path = folder_path + dataset_filename

    train_dataset, test_dataset, train_features, test_features, train_labels, test_labels, duration, n_nodes = load_dataset(
        complete_path,
        features,
        labels,
        slice_data=slice_data
    )

    model, history = load_model(model_path_filename, history_path_filename)

    if (model == None and history == None):

        earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
        callbacks = [earlystop]


        model = create_regression_nn_model(train_features,n_nodes)
        history = perform_neural_network_fit(model, train_features, train_labels, epochs, batch_size, validation_split, callbacks, verbose=1)

        if(save_model_bool):
            save_model(model, history, model_path_filename, history_path_filename)

    if(evaluate_model):
        evaluate_regression_nn_after_fit(model, test_features, test_labels)

    return model, history

def create_or_load_nn_classifier(folder_path, dataset_filename_no_leaks, dataset_filename_with_leaks, epochs, features, labels, batch_size=None,
                                model_path_filename="", history_path_filename="", slice_data=0.8,
                                validation_split=0.2, save_model_bool=False, fresh_start=False, evaluate_model=True):

    if(fresh_start):
        clean_old_models(model_path_filename)

    first_half_dataset_no_leaks, snd_half_dataset_no_leaks = process_dataset(folder_path,dataset_filename_no_leaks,None)
    first_half_dataset_with_leaks, snd_half_dataset_with_leaks = process_dataset(folder_path,dataset_filename_with_leaks,None)

    first_mixed_dataset = pd.concat([first_half_dataset_no_leaks,first_half_dataset_with_leaks], ignore_index=True)
    snd_mixed_dataset = pd.concat([snd_half_dataset_no_leaks, snd_half_dataset_with_leaks], ignore_index=True)

    train_features = first_mixed_dataset
    train_labels = train_features.pop(labels)

    test_features = snd_mixed_dataset
    test_labels = test_features.pop(labels)

    model, history = load_model(model_path_filename, history_path_filename)

    if (model == None and history == None):

        earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
        callbacks = [earlystop]

        # n_nodes = count_nodes_from_dataframe()

        model = create_classifier_nn_model(train_features,train_features.columns.size)
        history = perform_neural_network_fit(model, train_features, train_labels, epochs, batch_size, validation_split, callbacks, verbose=1)

        if(save_model_bool):
            save_model(model, history, model_path_filename, history_path_filename)

    if (evaluate_model):
        evaluate_regression_nn_after_fit(model, test_features, test_labels)

    return model, history


if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)
    print('Keras ', tf.keras.__version__)
    is_gpu_supported()

    # folder_path = "tensorflow_datasets/one_res_small/merged_2023/"
    #
    # dataset_filename = '1W_one_res_small_no_leaks_rand_bd_merged.csv'
    #
    # input_stat_full_dataset = "1W_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"
    #
    # # input_alt_dataset = ["1W_ALT_one_res_small_with_1_leaks_rand_base_dem_nodes_output.csv",
    # #                      "1W_ALT_one_res_small_with_1_at_8_leaks_rand_base_dem_nodes_output.csv",
    # #                      "1W_ALT_one_res_small_with_1_at_4_leaks_rand_base_dem_nodes_output.csv",
    # #                      "1W_ALT_one_res_small_with_1_at_2_leaks_rand_base_dem_nodes_output.csv"
    # #                      ]
    #
    # epochs = 1000
    # batch_size = count_nodes_from_dataframe(pd.read_csv(folder_path+dataset_filename))
    #
    # features = ["pressure_value", "base_demand"]
    # labels = "demand_value"
    #
    # model_path_filename = "tensorflow_models/regression_demand_model_1W"
    # history_path_filename = "regression_history_model"
    #
    # # This float indicates how much of the dataset will be used for training a new model (the rest will be used as a test dataset)
    # slice_data = 0.8
    #
    # # This bool will delete previous saved models with the same filenames indicated above
    # fresh_start = True
    #
    # # This bool will execute an evaluation after fitting the model
    # evaluate_model = True
    #
    # # This bool will determine if the (new) fitted model will be saved to the path and names indicated above
    # save_model_bool = True
    #
    # # tf.random.set_seed(2023)
    #
    # model, history = create_or_load_nn_regressor(folder_path, dataset_filename, epochs, features, labels,
    #                                              batch_size, model_path_filename=model_path_filename,
    #                                              history_path_filename=history_path_filename, slice_data=slice_data,
    #                                              fresh_start=fresh_start, evaluate_model=evaluate_model, save_model_bool=save_model_bool)

    process_folder = "tensorflow_datasets/one_res_small/gabriele_marzo_2023/"
    process_filename_with_leaks = "1M_one_res_small_fixed_leaks_rand_bd_filtered_merged.csv"

    process_filename_no_leaks = "1M_one_res_small_no_leaks_rand_bd_filtered_merged.csv"

    epochs = 1000
    batch_size = 10

    # features = ["pressure_value", "base_demand"]
    labels = "has_leak"

    model_path_filename = "tensorflow_models/classification_leak_processed_model_1M"
    history_path_filename = "classification_history_model"

    # This float indicates how much of the dataset will be used for training a new model (the rest will be used as a test dataset)
    slice_data = 0.8

    # This bool will delete previous saved models with the same filenames indicated above
    fresh_start = True

    # This bool will execute an evaluation after fitting the model
    evaluate_model = True

    # This bool will determine if the (new) fitted model will be saved to the path and names indicated above
    save_model_bool = True

    create_or_load_nn_classifier(process_folder, process_filename_no_leaks, process_filename_with_leaks, epochs,
                                 None, labels,model_path_filename=model_path_filename, history_path_filename=history_path_filename)





