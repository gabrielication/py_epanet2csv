import numpy as np
import pandas as pd
import os
import shutil
import csv
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
# import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers

# from keras_visualizer import visualizer

def clean_old_files():
    print("FRESH START ENABLED. Cleaning ALL old models and their files...")

    for filename in Path("..").glob("*.png"):
        try:
            os.remove(filename)
            print(str(filename) + " deleted")
        except OSError:
            print("\nError while deleting " + str(filename) + "\n")

    for filename in Path("..").glob("my_model"):
        try:
            shutil.rmtree(filename)

            print(str(filename) + " deleted")
        except OSError:
            print("\nError while deleting " + str(filename) + "\n")

    for filename in Path("..").glob("my_history.npy"):
        try:
            os.remove(filename)
            print(str(filename) + " deleted")
        except OSError:
            print("\nError while deleting " + str(filename) + "\n")

    print("All old files deleted.\n")

def fit_and_or_load_model(train_features, train_labels, epochs, validation_split, batch_size, callbacks, complete_path_stat, save_model=True, visualize_model_bool=True):
    input_filename_full_fitted_model = ""

    for filename in Path("..").glob("my_model"):
        input_filename_full_fitted_model = str(filename)

        # cheap hack. we just have one model file. break at first finding.
        break

    if input_filename_full_fitted_model != "":
        print("Full fitted model already exists. Loading " + input_filename_full_fitted_model + "...")
        # model_fitted = load(input_filename_full_fitted_model)

        model = tf.keras.models.load_model('my_model')

        history = np.load('my_history.npy', allow_pickle='TRUE').item()

        return model, history
    else:
        # we first fit the model on the complete dataset and save the fitted model back
        print("No old model found. Creating and Fitting...")

        model = create_neural_network_model(train_features, complete_path_stat, normalize=True)

        history = perform_neural_network_fit(model, train_features, train_labels, epochs,
                                             validation_split=validation_split, batch_size=batch_size,
                                             callbacks=callbacks, verbose=1)

        if(save_model):
            np.save('my_history.npy', history.history)

            output_filename_full_fitted_model = 'my_model'
            model.save(output_filename_full_fitted_model)

            print("Model saved to: " + output_filename_full_fitted_model)
        else:
            print("Model and History NOT SAVED!")


        if(visualize_model_bool):
            visualize_model(model)
        else:
            print("Model NOT VISUALIZED!")

        return model, history

def formatted_datetime():
    # current date and time
    now = str(datetime.now())
    now = now.replace(" ", "_")
    now = now.replace(".", "_")
    now = now.replace(":", "_")

    return now

def visualize_model(model):
    print("Generating model visualization...")

    now = formatted_datetime()

    output_filename = "model_simple_"+now+".png"

    tf.keras.utils.plot_model(model, to_file=output_filename, show_shapes=True)

    print(output_filename + " saved.")

    try:
        output_filename = "model_graph_" + now + ".png"

        # keras_visualizer will work only with normalizer disabled
        visualizer(model, format='png', view=True, filename=output_filename)

        print(output_filename + " saved.")
    except:
        print("PNG for keras_visualizer not saved! works only without Normalization layer.")

def is_gpu_supported():
    gpu_list = tf.config.list_physical_devices('GPU')

    if(len(gpu_list) == 0):
        print("GPU IS NOT SUPPORTED/ACTIVE/DETECTED!")

        return False
    else:
        print("GPU SUPPORTED: ",gpu_list)

        return True

def load_dataset(complete_path, cols, complete_path_stat, scaling=False, pairplot=False):
    print("LOADING " + complete_path + "...")

    # We read our entire dataset
    data = pd.read_csv(complete_path)

    # We drop these columns because they are strings, booleans and other incompatible types. We will convert them later

    print("Extracting only columns: ",cols)

    data_trans = data[cols].copy()

    # # Convert the types of the desired columns and add them back
    # le = preprocessing.LabelEncoder()
    # data_trans["hour"] = le.fit_transform(data["hour"])
    # data_trans["nodeID"] = le.fit_transform(data["nodeID"])
    # data_trans["node_type"] = le.fit_transform(data["node_type"])
    # data_trans["has_leak"] = le.fit_transform(data["has_leak"])

    data_scaled = data_trans

    if(scaling):
        scaler = StandardScaler()

        print("Standard Scaling IS ACTIVE. Preprocessing...")
        data_scaled = scaler.fit_transform(data_trans)
        data_scaled = pd.DataFrame(data_scaled, columns=[cols])

        print(data_trans.head())
        print(data_scaled.head())
        print("Preprocessing done.\n")
    else:
        print("Standard Scaling IS NOT ACTIVE.")

    print("Dividing FEATURES and LABELS...")

    # This was used in Tensorflow wiki but it's not the same as train test split. It will pick a SAMPLE jumping rows, not a clean SPLIT
    # train_dataset = data_scaled.sample(frac=0.8, random_state=0)

    df = pd.read_csv(complete_path_stat)
    n_nodes = int(df['number_of_nodes'].iloc[0])
    duration = int(df['time_spent_on_sim'].iloc[0])

    dataset_size = duration * n_nodes

    duration_percentage = int(0.8 * duration)

    train_dataset_size = duration_percentage * n_nodes
    test_dataset_size = duration - train_dataset_size

    train_dataset = data_scaled.iloc[:train_dataset_size, :]
    test_dataset = data_scaled.drop(train_dataset.index)

    if(pairplot):
        now = formatted_datetime()
        output_filename = "pairplot_"+now+".png"

        sns.pairplot(train_dataset[["pressure_value", "base_demand", "demand_value"]], diag_kind='kde').savefig(output_filename)

        print(output_filename+" saved.")

    # Tensorflow guide (https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb#scrollTo=2l7zFL_XWIRu)
    # says that the features are the columns that we want our network to train and labels is the value(s) to predict
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    # These instructions modificate also original dataframes
    train_labels = train_features.pop('demand_value')
    test_labels = test_features.pop('demand_value')

    return train_dataset, test_dataset, train_features, test_features, train_labels, test_labels

def create_neural_network_model(train_features, complete_path_stat, normalize=False):
    print("Building Neural Network Model...")

    if(normalize):
        print("NORMALIZATION IS ENABLED!")
        # We want to Normalize (scale) the data since it can be too different in ranges
        # These lines will create a NORMALIZATION layer (TODO: cerca) adapted to our data

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
    else:
        print("NORMALIZATION IS DISABLED!")

        feat_shape = train_features.shape[1]

        input_layer = layers.Input(shape=(feat_shape,))

    # These lines will just calculate the levels for the Deep Neural Net
    df = pd.read_csv(complete_path_stat)
    n_junc = int(df['number_of_junctions'].iloc[0])

    fst_level = n_junc * 5
    snd_level = n_junc * 3
    trd_level = n_junc

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
    # lossfn = 'mean_absolute_error'

    # opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    opt = tf.keras.optimizers.Adam()

    # metrics = ['mse', 'mae', tfa.metrics.r_square.RSquare()]
    metrics = ['mse', 'mae']

    #metrics = []

    # In regression we use mserr as loss funct
    # Adam is a good optimizer since it just do the calculation for SGD's parameters automatically starting from a fixed value
    model.compile(loss=lossfn,
                  metrics=metrics,
                  optimizer=opt)

    # Can be skipped
    model.summary()

    return model

def perform_neural_network_fit(model, train_features, train_labels, epochs, batch_size=None, validation_split=0.0, callbacks=[None], verbose = 1):
    # This array saves all the values obtained through the epochs

    print("epochs: ",epochs,"batch_size: ",batch_size, "validation_split: ", validation_split)

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

def plot_fit_results(history):
    # Plot results
    plt.clf()

    #TODO: epochs int values should stay on the x-axes

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    # Get the y limits
    loss_min, loss_max = min(history.history['loss']), max(history.history['loss'])
    val_loss_min, val_loss_max = min(history.history['loss']), max(history.history['loss'])

    ymin = min([loss_min,val_loss_min])
    ymax = max([loss_max, val_loss_max])

    # Set the y limits making the maximum 5% greater
    # plt.ylim(ymin, 1.05 * ymax)

    plt.xlabel('Epoch')
    plt.ylabel('Error [demand_value]')
    plt.legend()
    plt.grid(True)

    now = formatted_datetime()
    output_filename = "loss_plot_"+now+".png"

    plt.savefig(output_filename)

    print(output_filename+" saved.")

def evaluate_network_after_fit(model, test_features, test_labels):
    print("Evaluation started...")

    evl = model.evaluate(test_features, test_labels, verbose=0)

    loss = evl[0]
    mse = evl[1]
    mae = evl[2]
    r_square = None #evl[3]

    print("loss: ",loss)
    print("mse: ",mse)
    print("mae: ",mae)
    print("r_square: ",r_square)

    return loss, mse, mae, r_square

def predict_and_collect_results(model, test_features):
    print("Prediction started...")

    test_predictions = model.predict(test_features).flatten()

    return test_predictions

def run_predict_analysis(complete_path, complete_path_stat, epochs, cols, batch_size=None):
    print("PREDICT ANALYSIS:\n")

    validation_split = 0.2
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    callbacks = [earlystop]

    train_dataset, test_dataset, train_features, test_features, train_labels, test_labels = load_dataset(complete_path,
                                                                                                         cols,
                                                                                                         complete_path_stat,
                                                                                                         scaling=False,
                                                                                                         pairplot=False)

    model, history = fit_and_or_load_model(train_features, train_labels, epochs, validation_split, batch_size,
                                           callbacks, complete_path_stat, save_model=False, visualize_model_bool=False)

    evaluate_network_after_fit(model,test_features,test_labels)
    test_predictions = predict_and_collect_results(model, test_features)

    # plot_predictions(test_predictions, test_labels)

    return test_predictions, test_labels

def run_evaluation_analysis(complete_path, complete_path_stat, epochs, cols, batch_size=None):
    print("EVALUATION ANALYSIS:\n")

    validation_split = 0.2
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    callbacks = [earlystop]

    train_dataset, test_dataset, train_features, test_features, train_labels, test_labels = load_dataset(complete_path,cols,complete_path_stat,scaling=False, pairplot=False)

    model, history = fit_and_or_load_model(train_features, train_labels, epochs, validation_split, batch_size, callbacks, complete_path_stat)

    # last_fit_loss = history

    fit_loss = 0
    fit_mse = 0
    fit_mae = 0
    fit_r_square = 0

    fit_val_loss = 0
    fit_val_mse = 0
    fit_val_mae = 0
    fit_val_r_square = 0

    try:
        fit_loss = history.history['loss'][-1]
        fit_mse = history.history['mse'][-1]
        fit_mae = history.history['mae'][-1]
        fit_r_square = history.history['r_square'][-1]

        fit_val_loss = history.history['val_loss'][-1]
        fit_val_mse = history.history['val_mse'][-1]
        fit_val_mae = history.history['val_mae'][-1]
        fit_val_r_square = history.history['val_r_square'][-1]
    except:
        print("history is a dict.")

    try:
        plot_fit_results(history)
    except:
        print("don't need to print fit loss.")

    loss, mse, mae, r_square = evaluate_network_after_fit(model,test_features,test_labels)

    print("Done.")

    stop = earlystop.stopped_epoch

    print("STOP : ", stop)
    print()

    return loss, mse, mae, r_square, stop, fit_loss, fit_mse, fit_mae, fit_r_square, fit_val_loss, fit_val_mse, fit_val_mae, fit_val_r_square

def write_to_analysis_report_csv(X, writer, loss, mse, mae, r_square, fit_loss, fit_mse, fit_mae, fit_r_square, fit_val_loss, fit_val_mse, fit_val_mae,
                                 fit_val_r_square, delta_loss, delta_mse, delta_mae, delta_r_square, delta_fit_loss, delta_fit_mse,
                                 delta_fit_mae, delta_fit_r_square, delta_fit_val_loss, delta_fit_val_mse,
                                 delta_fit_val_mae, delta_fit_val_r_square, input_full_dataset, stop):

    base_demand = head_value = pressure_value = x_pos = y_pos = leak_area_value = leak_discharge_value = current_leak_demand_value = False
    smart_sensor_is_present = tot_network_demand = hour = nodeID = node_type = has_leak = False

    if "base_demand" in X:
        base_demand = True
    if "head_value" in X:
        head_value = True
    if "pressure_value" in X:
        pressure_value = True
    if "x_pos" in X:
        x_pos = True
    if "y_pos" in X:
        y_pos = True
    if "leak_area_value" in X:
        leak_area_value = True
    if "leak_discharge_value" in X:
        leak_discharge_value = True
    if "current_leak_demand_value" in X:
        current_leak_demand_value = True
    if "smart_sensor_is_present" in X:
        smart_sensor_is_present = True
    if "tot_network_demand" in X:
        tot_network_demand = True
    if "hour" in X:
        hour = True
    if "nodeID" in X:
        nodeID = True
    if "node_type" in X:
        node_type = True
    if "has_leak" in X:
        has_leak = True

    #short CSV
    out_row = [base_demand, pressure_value, loss, mse, mae, r_square, fit_loss,
              fit_mse, fit_mae, fit_r_square, fit_val_loss, fit_val_mse, fit_val_mae, fit_val_r_square,
              delta_loss, delta_mse, delta_mae, delta_r_square, delta_fit_loss, delta_fit_mse,
              delta_fit_mae, delta_fit_r_square, delta_fit_val_loss, delta_fit_val_mse,
              delta_fit_val_mae, delta_fit_val_r_square,
              input_full_dataset,stop]

    writer.writerow(out_row)


def create_analysis_report(folder_input, input_full_dataset, input_list_of_alt_datasets, input_stat_full_dataset, cols, label, epochs, fresh_start=False):

    now = formatted_datetime()

    output_filename = input_full_dataset[0:3]+"tensorflow_report_"+now+".csv"

    # open the file in the write mode
    f = open(output_filename, "w", newline='', encoding='utf-8')

    # create the csv writer
    writer = csv.writer(f)

    header = ["base_demand", "pressure_value", "loss", "mse", "mae", "r_square", "fit_loss",
              "fit_mse", "fit_mae", "fit_r_square", "fit_val_loss", "fit_val_mse", "fit_val_mae", "fit_val_r_square",
              "delta_loss", "delta_mse", "delta_mae", "delta_r_square", "delta_fit_loss", "delta_fit_mse",
              "delta_fit_mae", "delta_fit_r_square", "delta_fit_val_loss", "delta_fit_val_mse",
              "delta_fit_val_mae", "delta_fit_val_r_square",
              "dataset","epochs"]

    writer.writerow(header)

    for i in range(len(cols)):
        oc = combinations(cols, i + 1)
        for c in oc:
            if (fresh_start):
                clean_old_files()

            new_cols = list(c)
            new_cols.append(label)

            # if(len(new_cols) != 3):
            #     break
            # else:
            #     print("GOOD LEN")

            #print(new_cols)

            complete_path = folder_input + input_full_dataset
            complete_path_stat = folder_input + input_stat_full_dataset

            loss, mse, mae, r_square, stop, fit_loss, fit_mse, fit_mae, fit_r_square, fit_val_loss,\
            fit_val_mse, fit_val_mae, fit_val_r_square = run_evaluation_analysis(complete_path, complete_path_stat, epochs, new_cols)

            write_to_analysis_report_csv(new_cols, writer, loss, mse, mae, r_square, fit_loss, fit_mse, fit_mae,
                                         fit_r_square, fit_val_loss, fit_val_mse, fit_val_mae, fit_val_r_square,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, input_full_dataset, stop)

            ##########

            for input_alt_dataset in input_list_of_alt_datasets:

                complete_path = folder_input + input_alt_dataset
                complete_path_stat = folder_input + input_stat_full_dataset

                alt_loss, alt_mse, alt_mae, alt_r_square, alt_stop, alt_fit_loss, alt_fit_mse, alt_fit_mae, alt_fit_r_square, \
                alt_fit_val_loss, alt_fit_val_mse, alt_fit_val_mae, alt_fit_val_r_square = run_evaluation_analysis(complete_path,
                                                                                                                   complete_path_stat,
                                                                                                                   epochs,
                                                                                                                   new_cols)

                delta_loss = alt_loss - loss
                delta_mse = alt_mse - mse
                delta_mae = alt_mae - mae
                delta_r_square = alt_r_square - r_square

                delta_fit_loss = alt_loss - fit_loss
                delta_fit_mse = alt_mse - fit_mse
                delta_fit_mae = alt_mae - fit_mae
                delta_fit_r_square = alt_r_square - fit_r_square

                delta_fit_val_loss = alt_loss - fit_val_loss
                delta_fit_val_mse = alt_mse - fit_val_mse
                delta_fit_val_mae = alt_mae - fit_val_mae
                delta_fit_val_r_square = alt_r_square - fit_val_r_square

                write_to_analysis_report_csv(new_cols, writer, alt_loss, alt_mse, alt_mae, alt_r_square, alt_fit_loss, alt_fit_mse,
                                             alt_fit_mae, alt_fit_r_square, alt_fit_val_loss, alt_fit_val_mse, alt_fit_val_mae,
                                             alt_fit_val_r_square, delta_loss, delta_mse, delta_mae, delta_r_square, delta_fit_loss,
                                             delta_fit_mse,
                                             delta_fit_mae, delta_fit_r_square, delta_fit_val_loss, delta_fit_val_mse,
                                             delta_fit_val_mae, delta_fit_val_r_square, input_alt_dataset, alt_stop)

            # print()

    f.close()

def create_prediction_report(folder_input, input_full_dataset, input_list_of_alt_datasets,
                             input_stat_full_dataset, cols, label, epochs, fresh_start=False, batch_size=None):
    now = formatted_datetime()

    # output_filename = input_full_dataset[0:3] + "tensorflow_report_" + now + ".csv"
    output_filename = input_full_dataset[0:3] + "test_prediction_report_" + now + ".csv"

    # open the file in the write mode
    f = open(output_filename, "w", newline='', encoding='utf-8')

    # create the csv writer
    writer = csv.writer(f)

    header = ["predictions","true_test_values","error"]

    writer.writerow(header)

    complete_path = folder_input + input_full_dataset
    complete_path_stat = folder_input + input_stat_full_dataset

    if (fresh_start):
        clean_old_files()

    # new_cols = list(c) TODO: combination

    # Currently not doing any column combination
    new_cols = cols
    new_cols.append(label)

    test_predictions, test_labels = run_predict_analysis(complete_path,complete_path_stat,epochs,cols, batch_size=batch_size)

    for pred,test in zip(test_predictions,test_labels):
        output_row = [pred,test,pred-test]
        writer.writerow(output_row)

    f.close()

    print("\nPrediction report saved to: "+output_filename)
    print("Quitting...")

if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)
    print('Keras ', tf.keras.__version__)
    is_gpu_supported()

    folder_input = "tensorflow_datasets/"
    folder_network = "one_res_small/no_leaks_rand_base_demand/1W/"

    input_full_dataset = folder_network + '1W_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'
    input_stat_full_dataset = folder_network + "1W_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"

    input_alt_dataset = [folder_network + "1W_ALT_one_res_small_with_1_leaks_rand_base_dem_nodes_output.csv",
                         folder_network + "1W_ALT_one_res_small_with_1_at_8_leaks_rand_base_dem_nodes_output.csv",
                         folder_network + "1W_ALT_one_res_small_with_1_at_4_leaks_rand_base_dem_nodes_output.csv",
                         folder_network + "1W_ALT_one_res_small_with_1_at_2_leaks_rand_base_dem_nodes_output.csv"
                         ]

    epochs = 100
    batch_size = 83 #number of nodes

    cols = ["pressure_value", "base_demand"]
    label = "demand_value"

    # create_analysis_report(folder_input, input_full_dataset, input_alt_dataset, input_stat_full_dataset, cols, label,
    #                        epochs, fresh_start=True)

    create_prediction_report(folder_input, input_full_dataset, input_alt_dataset, input_stat_full_dataset, cols, label,
                             epochs, fresh_start=True, batch_size=batch_size)

    # folder_input = ""

    # input_full_dataset = '1W_Net3_no_leaks_nodes_output.csv'
    # input_stat_full_dataset = "1W_Net3_no_leaks_nodes_simulation_stats.csv"
    #

    # input_full_dataset = '1W_Net3_no_leaks_nodes_output.csv'
    # input_stat_full_dataset = "1W_Net3_no_leaks_nodes_simulation_stats.csv"
    #
    # input_alt_dataset = [
    #                      "1W_Net3_with_leaks_nodes_output.csv"
    #                      ]
    #

    #
    # print("FRESH START ENABLED. Cleaning ALL old models and their files...")
    # clean_old_files()
    #
    # complete_path = folder_input + input_full_dataset
    # complete_path_stat = folder_input + input_stat_full_dataset
    #
    # cols.append(label)
    #
    # run_evaluation_analysis(complete_path, complete_path_stat, epochs, cols)