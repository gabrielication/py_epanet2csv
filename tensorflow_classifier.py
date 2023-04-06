import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential

# Conv2D is a convolutional layer used for 2D spatial data,
# where the input data can be thought of as a grid or image.
# Each element in the grid (i.e., each pixel in the image) is
# typically represented by a single scalar value or a vector of values,
# depending on the specific problem. For example, in image classification,
# each pixel may be represented by a vector of RGB values, or a single
# grayscale value.
#
# On the other hand, Conv1D is a convolutional layer used for 1D sequential
# data, where the input data consists of a sequence of data points, such as
# a time series or a sequence of words. Each data point in the sequence may
# be represented by a scalar value or a vector of values, depending on the
# specific problem.
#
# In your case, your input data has the shape (670, 83, 8), which represents
# a 3D array with 670 samples, 83 time steps, and 8 features per time step.
# This is a typical representation of 1D sequential data, where each sample
# represents a sequence of 83 time steps, and each time step is represented by
# an 8-dimensional feature vector. Therefore, Conv1D is the appropriate
# convolutional layer to use for this type of data.
#
# If your input data were, for example, a set of 670 grayscale images,
# each with dimensions (83, 83), then you would use Conv2D instead. Each
# image would be represented by a 2D array with 83 rows and 83 columns,
# and the Conv2D layer would apply a 2D convolution operation to each
# image independently.

# In the context of your specific input data, the shape (670, 83, 8)
# means that you have 670 samples, each of which is a sequence of 83 time
# steps, and each time step is represented by an 8-dimensional feature vector.
#
# Here's an example to help illustrate this concept. Let's say you are predicting
# the temperature at a particular location over time, based on a set of 8 features
# (such as humidity, wind speed, etc.) that are measured at that location.
# You might collect data for a set of 670 days, with measurements taken every hour
# for each day. In this case, each sample corresponds to a single day, the 83 time
# steps correspond to the 24 hours in a day, and each time step is represented by
# an 8-dimensional vector of feature values. So for example, the first time step
# for the first day might look like (0.62, 3.4, 0.01, 0.12, 0.82, 0.11, 0.44, 0.09),
# where each value represents a different feature.
#
# Given this representation of the input data, Conv1D is an appropriate convolutional
# layer to use because it is specifically designed for processing sequential data with
# a time axis (such as time series or natural language text). In this case,
# Conv1D can be used to perform convolutions across the 83 time steps for each sample,
# taking into account the temporal dependencies between the data points.

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("tensorflow_datasets/one_res_small/gabriele_marzo_2023/1M_one_res_small_rand_leaks_rand_fixed_bd_with_multipliers_merged.csv")

columns = df[df["hour"] == "0:00:00"]["nodeID"].array

max_hour = int(df.iloc[-1]["hour"].split(":")[0])

out_dict = []

# TENSORFLOW ONLY ACCEPTS LISTS, not DF with lists!

for hour in range(3):
    timestamp = str(hour) + ":00:00"

    print("Processing timestamp: ", timestamp)

    temp2 = []
    for nodeID in columns:
        # print(timestamp, nodeID)

        temp = df[df["hour"] == timestamp]
        temp = temp[temp["nodeID"] == nodeID]

        base_demand = float(temp.iloc[-1]["base_demand"])
        demand_value = float(temp.iloc[-1]["demand_value"])
        head_value = float(temp.iloc[-1]["head_value"])
        pressure_value = float(temp.iloc[-1]["pressure_value"])
        x_pos = float(temp.iloc[-1]["x_pos"])
        y_pos = float(temp.iloc[-1]["y_pos"])

        has_leak = int(temp.iloc[-1]["has_leak"])

        output = [base_demand, demand_value, head_value, pressure_value, x_pos, y_pos]

        temp2.append(output)

    out_dict.append(temp2)

X = out_dict

y = [1,0,1]

# # Define the model architecture
# model = tf.keras.models.Sequential([
#   tf.keras.layers.LSTM(units=32, input_shape=(83, 6)),
#   tf.keras.layers.Dense(units=1, activation='sigmoid')
# ])

# Define the input shape
input_shape = (83, 6)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10)
