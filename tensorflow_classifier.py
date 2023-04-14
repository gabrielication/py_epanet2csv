import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import pandas as pd
import tensorflow_addons as tfa

df = pd.read_pickle("tensorflow_datasets/one_res_small/gabriele_marzo_2023/processed_df.pickle")

target = df.pop("has_leak")

data = np.array(df.values.tolist())

num_samples = data.shape[0]
num_features = data.shape[1]
num_channels = len(data[0][0])  # since each feature has 6 floats

# Reshape the input data into a 3D tensor
X = np.reshape(data, (num_samples, num_features, num_channels))

temp_lab = target.values.tolist()

y = np.reshape(temp_lab, (num_samples, num_features))

# Define the MLP model
model = Sequential([
    Flatten(input_shape=(num_features, num_channels)),
    BatchNormalization(),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(num_features, activation='sigmoid')
])

loss=tf.keras.losses.BinaryCrossentropy()

# accuracy measures the overall correctness of the model's predictions,
# precision measures how often the model is correct when it predicts a positive instance,
# recall measures how well the model can identify positive instances,
# F1 score combines precision and recall to give a single measure of the model's performance

# These are the metrics for a binary classification problem
metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=83)]

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss=loss, optimizer='adam', metrics=metrics)

# Train the model
model.fit(X, y, epochs=100, batch_size=4, validation_split=0.2, shuffle=False)