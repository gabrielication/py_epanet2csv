import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print('Tensorflow ',tf.__version__)
print('Keras ',tf.keras.__version__)

folder_input = "datasets_for_mlp/"

input_full_dataset = '1D_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'
input_stat_full_dataset = "1D_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"

complete_path = folder_input+input_full_dataset

complete_path_stat = folder_input+input_stat_full_dataset

print("LOADING " + complete_path + "...")

# We read our entire dataset
data = pd.read_csv(complete_path)

# We drop these columns because they are strings, booleans and other incompatible types. We will convert them later
data_trans = data.drop(columns=["hour", "nodeID", "node_type", "has_leak"])

# Convert the types of the desired columns and add them back
le = preprocessing.LabelEncoder()
data_trans["hour"] = le.fit_transform(data["hour"])
data_trans["nodeID"] = le.fit_transform(data["nodeID"])
data_trans["node_type"] = le.fit_transform(data["node_type"])
data_trans["has_leak"] = le.fit_transform(data["has_leak"])

print("Dividing X and y matrices...")

# This is basically train_test_split from sklearn...but directly from tensorflow!
train_dataset = data_trans.sample(frac=0.8, random_state=0)
test_dataset = data_trans.drop(train_dataset.index)

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

normalizer.adapt(np.array(train_features))

normalizer.mean.numpy()

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
      normalizer,
      layers.Dense(fst_level, activation='relu'),
      layers.Dense(snd_level, activation='relu'),
      layers.Dense(trd_level, activation='relu'),
      layers.Dense(1)
  ])

# In regression we use mserr as loss funct
# Adam is a good optimizer since it just do the calculation for SGD's parameters automatically starting from a fixed value
model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))

# Can be skipped
model.summary()

# This array saves all the values obtained through the epochs
history = model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=50)

# Plot results
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 0.2])
plt.xlabel('Epoch')
plt.ylabel('Error [demand_value]')
plt.legend()
plt.grid(True)
plt.savefig("provolone.png")

test_results = {}
test_results['model'] = model.evaluate(test_features, test_labels, verbose=0)

# print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

test_predictions = model.predict(test_features).flatten()
#
# plt.clf()
#
# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)
# plt.savefig("provola2.png")
