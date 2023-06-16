import numpy as np
import pandas as pd
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa
import visualkeras

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split

def is_gpu_supported():
    gpu_list = tf.config.list_physical_devices('GPU')

    if (len(gpu_list) == 0):
        print("GPU IS NOT SUPPORTED/ACTIVE/DETECTED!")

        return False
    else:
        print("GPU SUPPORTED: ", gpu_list)

        return True

def visualize_model(model_path_filename, history_path_filename):
    model, history = load_model(model_path_filename, history_path_filename)

    visualkeras.layered_view(model, legend=False, spacing=200, scale_xy=10).show()

def clean_old_models(model_path):
    print("FRESH START ENABLED. Cleaning ALL old models and their files...")

    for filename in Path(".").glob(model_path):
        try:
            shutil.rmtree(filename)

            print(str(filename) + " deleted")
        except OSError:
            print("\nError while deleting " + str(filename) + "\n")

    print("All old files deleted.\n")

def load_model(model_path_filename, history_path_filename):
    if os.path.exists(model_path_filename):
        print("Model already exists!\nIf tensorflow versions from saved one differ then a crash might happen!")
        model = tf.keras.models.load_model(model_path_filename)

        history_complete_path = model_path_filename + '/' + history_path_filename + '.npy'

        history = np.load(history_complete_path, allow_pickle='TRUE').item()

        print("Loading model: " + model_path_filename + "/\nLoading history: " + history_complete_path)

        return model, history
    else:
        print("Previous fitted model not found!")

        return None, None

def save_model(model, history, model_path_filename, history_path_filename):
    model.save(model_path_filename)

    history_complete_path = model_path_filename + '/' + history_path_filename + '.npy'

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

def obtain_features_and_labels(folder_path, filename):

    complete_path_filename = folder_path + filename

    df = pd.read_pickle(complete_path_filename)

    target = df.pop("demand_value")

    train_data = np.array(df.values.tolist())

    num_samples = train_data.shape[0]
    num_features = train_data.shape[1]
    num_channels = len(train_data[0][0])  # since each feature has 6 floats

    # Reshape the input data into a 3D tensor
    X = np.reshape(train_data, (num_samples, num_features, num_channels))

    temp_lab = target.values.tolist()

    y = np.reshape(temp_lab, (num_samples, num_features))

    return X, y, num_samples, num_features, num_channels

def nn_regressor(folder_path, filename, epochs, batch_size=None,
                  model_path_filename="", history_path_filename="",
                  validation_split=0.2, patience_early_stop=10, save_model_bool=False,
                  folder_path_val="", filename_val=""):
    print("\nNN Classifier launched!\n")

    print("Parameters:")
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("model_path_filename:", model_path_filename)
    print("history_path_filename:", history_path_filename)
    print("validation_split:", validation_split)
    print("save_model_bool:", save_model_bool)
    # print("patience_early_stop:", patience_early_stop)
    print()

    if (folder_path_val != "" and filename_val != ""):
        print("Validation set provided. Loading validation set...")

        X_train, y_train, num_samples_train, num_features_train, num_channels_train = obtain_features_and_labels(
            folder_path, filename)

        X_val, y_val, num_samples_val, num_features_val, num_channels_val = obtain_features_and_labels(folder_path_val, filename_val)

    else:
        print("No validation set provided. Splitting training set into training and validation sets...")

        X, y, num_samples_train, num_features_train, num_channels_train = obtain_features_and_labels(
            folder_path, filename)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

    # Preferire sempre una rete semplice!

    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(num_features_train, num_channels_train)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_features_train, activation='linear')
    ])

    loss = 'mean_squared_error'

    metrics = ['mse', 'mae']

    # Compile the model with binary crossentropy loss and Adam optimizer
    model.compile(loss=loss, optimizer='adam', metrics=metrics)

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    if (save_model_bool):
        print()

        now = formatted_datetime()
        save_model(model, history, model_path_filename + "_" + now, history_path_filename)

    # Evaluate the model on the test data using `evaluate`
    # print("Evaluate on test data")
    # results = model.evaluate(x_test, y_test, batch_size=128)
    # print("test loss, test acc:", results)

    return model, history

def evaluate_and_predict_leakages(X, y=None, model=None, history=None, load_model_bool=False, model_path="", history_path=""):
    model = model
    history = history

    if(load_model_bool):
        model,history = load_model(model_path, history_path)

    print("Evaluate on test data")
    results = model.evaluate(X, y, batch_size=128)
    print("test loss, test acc:", results)

    print("Generate predictions for 3 samples")
    predictions = model.predict(X[:10])

    print("predictions shape:", predictions.shape)

    output_csv_with_predictions_diff(y[:10], predictions)

def output_csv_with_predictions_diff(true_values, pred_values, output_filename):
    # Calculate the difference between array1 and array2
    difference = true_values - pred_values

    # Define the CSV file path
    csv_file = output_filename

    # Write the data to CSV
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["True Values", "Predictions", "Difference"])  # Write header

        for i in range(true_values.shape[0]):
            for j in range(true_values.shape[1]):
                true = format(true_values[i][j], '.4f').replace('.', ',')
                pred = format(pred_values[i][j], '.4f').replace('.', ',')
                diff = format(difference[i][j], '.4f').replace('.', ',')

                writer.writerow([true, pred, diff])

    print(f"CSV file '{csv_file}' created successfully.")

if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)
    print('Keras ', tf.keras.__version__)
    is_gpu_supported()

    folder_path = "tensorflow_datasets/8_juncs_1_res/"
    filename = "1Y_8_junctions_1_res_no_leaks_rand_bd_conv1d_regression.pickle"

    # folder_path_val = "tensorflow_datasets/one_res_small/gabriele_maggio_2023/"
    # filename_val = "conv1d_rand_leaks_each_sim_transposed_dataset.pickle"

    # Where to save/load the fitted model and its history file
    model_path_filename = "tensorflow_models/regression_1Y_processed"
    history_path_filename = "regression_history_model"

    # This bool will determine if the (new) fitted model will be saved to the path and names indicated above
    save_model_bool = True

    # epochs during fit
    epochs = 50

    # batch size to be used during fit
    batch_size = 32

    # This float will split the data for validation during fit
    validation_split = 0.2

    # This bool will add Dropouts layers to the NN
    dropout = True

    # This int determines how many epochs should we monitor before stopping fitting if the situation does not improve
    patience_early_stop = 100

    nn_regressor(folder_path, filename, epochs, batch_size=batch_size,
                  model_path_filename=model_path_filename, history_path_filename=history_path_filename,
                  validation_split=validation_split, patience_early_stop=patience_early_stop,
                  save_model_bool=save_model_bool)

    # folder_path = "tensorflow_datasets/8_juncs_1_res/"
    # filename = "1M_8_junctions_1_res_no_leaks_rand_bd_regression_validation.pickle"

    # X, y, num_samples, num_features, num_channels = obtain_features_and_labels(folder_path, filename)
    # #
    # model_path_filename = "tensorflow_models/regression_1Y_processed_2023-06-07_16_47_27_362593"
    # #
    # evaluate_and_predict_leakages(X, y, load_model_bool=True, model_path=model_path_filename, history_path=history_path_filename)

    # folder_path = "tensorflow_datasets/8_juncs_1_res/"
    # filename = "1M_8_junctions_1_res_with_1_leak_rand_bd_validation.pickle"
    # #
    # X, y, num_samples, num_features, num_channels = obtain_features_and_labels(folder_path, filename)
    # #
    # model_path_filename = "tensorflow_models/regression_1Y_processed_2023-06-07_16_47_27_362593"
    # #
    # evaluate_and_predict_leakages(X, y, load_model_bool=True, model_path=model_path_filename, history_path=history_path_filename)

