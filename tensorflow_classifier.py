import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split

# from keras_visualizer import visualizer

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

    target = df.pop("has_leak")

    train_data = np.array(df.values.tolist())

    num_samples = train_data.shape[0]
    num_features = train_data.shape[1]
    num_channels = len(train_data[0][0])  # since each feature has 6 floats

    # Reshape the input data into a 3D tensor
    X = np.reshape(train_data, (num_samples, num_features, num_channels))

    temp_lab = target.values.tolist()

    y = np.reshape(temp_lab, (num_samples, num_features))

    return X, y, num_samples, num_features, num_channels

def nn_classifier(folder_path, filename, epochs, batch_size=None,
                  model_path_filename="", history_path_filename="",
                  validation_split=0.2, patience_early_stop=10, save_model_bool=False,
                  folder_path_val="", filename_val=""):
    print("\nNN Classifier launched!\n")

    print("Parameters:")
    print("epochs:", epochs)
    print("batch_size:", batch_size)
    print("model_path_filename:", model_path_filename)
    print("history_path_filename:", history_path_filename)
    # print("validation_split:", validation_split)
    print("save_model_bool:", save_model_bool)
    # print("patience_early_stop:", patience_early_stop)
    print()

    if (folder_path_val != "" and filename_val != ""):
        X_train, y_train, num_samples_train, num_features_train, num_channels_train = obtain_features_and_labels(
            folder_path, filename)

        X_val, y_val, num_samples_val, num_features_val, num_channels_val = obtain_features_and_labels(folder_path_val, filename_val)

    else:
        X, y, num_samples_train, num_features_train, num_channels_train = obtain_features_and_labels(
            folder_path, filename)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preferire sempre una rete semplice!
    # Dropout 0.9 - richiederà almeno 1000 (?) epochs. scaliamo a 0.8, 0.7... fino a che converge

    model = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(input_shape=(num_features_train, num_channels_train)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_features_train, activation='sigmoid')
    ])

    loss = tf.keras.losses.BinaryCrossentropy()

    # accuracy is a metric that calculates the fraction of correctly classified samples over the total number of samples
    # binary accuracy calculates how often predictions match binary labels
    # precision measures how often the model is correct when it predicts a positive instance,
    # recall measures how well the model can identify positive instances,
    # F1 score combines precision and recall to give a single measure of the model's performance

    # These are the metrics for a binary classification problem
    # metrics = ['accuracy', tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=83)]
    metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision()]

    # Compile the model with binary crossentropy loss and Adam optimizer
    model.compile(loss=loss, optimizer='adam', metrics=metrics)

    # earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_early_stop)
    # callbacks = [earlystop]

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
    predictions = model.predict(X[:3])
    predictions_binary = np.where(predictions > 0.5, 1, 0)

    print("predictions shape:", predictions.shape)


if __name__ == "__main__":
    print('Tensorflow ', tf.__version__)
    print('Keras ', tf.keras.__version__)
    is_gpu_supported()

    folder_path = "tensorflow_datasets/one_res_small/gabriele_maggio_2023/"
    filename = "1M_conv1d_rand_leaks_rand_bd_transposed_dataset.pickle"

    folder_path_val = "tensorflow_datasets/one_res_small/gabriele_maggio_2023/"
    filename_val = "conv1d_rand_leaks_each_sim_transposed_dataset.pickle"

    # Where to save/load the fitted model and its history file
    model_path_filename = "tensorflow_models/classification_12M_processed"
    history_path_filename = "classification_history_model"

    # This bool will determine if the (new) fitted model will be saved to the path and names indicated above
    save_model_bool = True

    # epochs during fit
    epochs = 1000

    # batch size to be used during fit
    batch_size = 32

    # This float will split the data for validation during fit
    validation_split = 0.2

    # This bool will add Dropouts layers to the NN
    dropout = True

    # This int determines how many epochs should we monitor before stopping fitting if the situation does not improve
    patience_early_stop = 100

    # nn_classifier(folder_path, filename, epochs, batch_size=batch_size,
    #               model_path_filename=model_path_filename, history_path_filename=history_path_filename,
    #               validation_split=validation_split, patience_early_stop=patience_early_stop,
    #               save_model_bool=save_model_bool, folder_path_val=folder_path_val, filename_val=filename_val)

    folder_path = "tensorflow_datasets/one_res_small/gabriele_maggio_2023/"
    filename = "1M_conv1d_rand_leaks_rand_bd_transposed_dataset.pickle"

    X, y, num_samples, num_features, num_channels = obtain_features_and_labels(folder_path, filename)

    model_path_filename = "tensorflow_models/91ACCURACY_classification_1Y_processed_2023-05-17_18_28_31_052484"

    evaluate_and_predict_leakages(X, y, load_model_bool=True, model_path=model_path_filename, history_path=history_path_filename)