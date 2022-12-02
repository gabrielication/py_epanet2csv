import numpy as np

from sklearn.preprocessing import StandardScaler

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa

from keras_visualizer import visualizer

from tensorflow import keras
from tensorflow.keras import layers

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
    else:
        print("GPU SUPPORTED: ",gpu_list)

def load_dataset(complete_path, cols, scaling=False, pairplot=False):
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
    # This is basically train_test_split from sklearn...but directly from tensorflow!
    train_dataset = data_scaled.sample(frac=0.8, random_state=0)
    test_dataset = data_scaled.drop(train_dataset.index)

    # sns.pairplot(train_dataset[["pressure_value", "base_demand", "demand_value"]], diag_kind='kde').savefig("pairplot.png")

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

    metrics = ['mse', 'mae', tfa.metrics.r_square.RSquare()]

    #metrics = []

    # In regression we use mserr as loss funct
    # Adam is a good optimizer since it just do the calculation for SGD's parameters automatically starting from a fixed value
    model.compile(loss=lossfn,
                  metrics=metrics,
                  optimizer=opt)

    # Can be skipped
    model.summary()

    return model

def perform_neural_network_fit(model, train_features, train_labels, epochs, batch_size=None, validation_split=0.0, callbacks=None):
    # This array saves all the values obtained through the epochs

    print("epochs: ",epochs,"batch_size: ",batch_size, "validation_split: ", validation_split)

    history = model.fit(
        train_features,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )

    return history

def plot_fit_results(history):
    # Plot results
    plt.clf()

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    # Get the y limits
    ymin, ymax = min(history.history['loss']), max(history.history['loss'])

    # Set the y limits making the maximum 5% greater
    plt.ylim(ymin, 1.05 * ymax)

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
    r_square = evl[3]

    print("loss: ",loss)
    print("mse: ",mse)
    print("mae: ",mae)
    print("r_square: ",r_square)

    return loss, mse, mae, r_square

def run_analysis(complete_path, complete_path_stat, epochs):
    print('Tensorflow ', tf.__version__)
    print('Keras ', tf.keras.__version__)
    is_gpu_supported()

    validation_split = 0.2
    batch_size = None
    # not implemented callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    cols = ["pressure_value", "base_demand", "demand_value"]

    train_dataset, test_dataset, train_features, test_features, train_labels, test_labels = load_dataset(complete_path,cols,scaling=False, pairplot=False)

    model = create_neural_network_model(train_features, complete_path_stat, normalize=True)

    history = perform_neural_network_fit(model,train_features, train_labels, epochs,
                                         validation_split=validation_split, batch_size=batch_size)

    plot_fit_results(history)

    visualize_model(model)

    loss, mse, mae, r_square = evaluate_network_after_fit(model,test_features,test_labels)

    print("Done.")

if __name__ == "__main__":
    folder_input = "datasets_for_mlp/"

    input_full_dataset = '1D_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'
    input_stat_full_dataset = "1D_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"

    complete_path = folder_input + input_full_dataset

    complete_path_stat = folder_input + input_stat_full_dataset

    run_analysis(complete_path, complete_path_stat, 5)