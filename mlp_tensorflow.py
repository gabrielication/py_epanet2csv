import numpy as np

from sklearn.preprocessing import StandardScaler

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print('Tensorflow ',tf.__version__)
print('Keras ',tf.keras.__version__)
print("GPU SUPPORT: ",tf.config.list_physical_devices('GPU'))

folder_input = "datasets_for_mlp/"

input_full_dataset = '1W_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'
input_stat_full_dataset = "1W_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"

complete_path = folder_input+input_full_dataset

complete_path_stat = folder_input+input_stat_full_dataset

print("LOADING " + complete_path + "...")

# We read our entire dataset
data = pd.read_csv(complete_path)

# We drop these columns because they are strings, booleans and other incompatible types. We will convert them later

# data_trans = data.drop(columns=["hour", "nodeID", "node_type", "has_leak"])
cols = ["pressure_value", "base_demand", "demand_value","x_pos","y_pos","tot_network_demand"]

data_trans = data[cols].copy()


# # Convert the types of the desired columns and add them back
# le = preprocessing.LabelEncoder()
# data_trans["hour"] = le.fit_transform(data["hour"])
# data_trans["nodeID"] = le.fit_transform(data["nodeID"])
# data_trans["node_type"] = le.fit_transform(data["node_type"])
# data_trans["has_leak"] = le.fit_transform(data["has_leak"])

data_scaled = data_trans

#scaler = StandardScaler()

#print("Standard Scaling Data...")
#data_scaled = scaler.fit_transform(data_trans)
#data_scaled = pd.DataFrame(data_scaled, columns = [cols])

# print(data_trans.head())
# print(data_scaled.head())
#
# exit()

print("Dividing X and y matrices...")
# This is basically train_test_split from sklearn...but directly from tensorflow!
train_dataset = data_scaled.sample(frac=0.8, random_state=0)
test_dataset = data_scaled.drop(train_dataset.index)

# Tensorflow guide (https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb#scrollTo=2l7zFL_XWIRu)
# says that the features are the columns that we want our network to train and labels is the value(s) to predict
train_features = train_dataset.copy()
test_features = test_dataset.copy()

# These instructions modificate also original dataframes
train_labels = train_features.pop('demand_value')
test_labels = test_features.pop('demand_value')

# print(train_dataset.describe().transpose()[['mean', 'std']])

# We want to Normalize (scale) the data since it can be too different in ranges
# These lines will create a NORMALIZATION layer (TODO: cerca) adapted to our data
normalizer = tf.keras.layers.Normalization(axis=-1)
#
normalizer.adapt(np.array(train_features))
#
normalizer.mean.numpy()

# These lines will just calculate the levels for the Deep Neural Net
df = pd.read_csv(complete_path_stat)
n_junc = int(df['number_of_junctions'].iloc[0])

fst_level = n_junc * 5
snd_level = n_junc * 3
trd_level = n_junc

print(data_scaled.shape)

# exit()

# Let's build the model. The first layer will be the normalizer that we built before
# Depth=3 and Width=fst,snd,trd
# Since it is a Regression problem, last one will be a Linear output (funzione immagine)
model = keras.Sequential([
      normalizer,
      #layers.Dense(fst_level, activation='relu', input_dim=train_features.shape[1]),
      layers.Dense(fst_level, activation='relu'),
      layers.Dense(snd_level, activation='relu'),
      layers.Dense(trd_level, activation='relu'),
      layers.Dense(1)
  ])

lossfn = 'mean_squared_error'
# lossfn = 'mean_absolute_error'

# opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
opt = tf.keras.optimizers.Adam()

metrics = ['mse', 'mae']

# In regression we use mserr as loss funct
# Adam is a good optimizer since it just do the calculation for SGD's parameters automatically starting from a fixed value
model.compile(loss=lossfn,
              metrics=metrics,
                optimizer=opt)

# Can be skipped
model.summary()

# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

batch_size = 16

# This array saves all the values obtained through the epochs
history = model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    #batch_size=batch_size,
    # callbacks=[callback],
    epochs=3)

# Plot results
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
plt.savefig("loss_plot.png")

evl = model.evaluate(test_features, test_labels, verbose=0)

print(evl)