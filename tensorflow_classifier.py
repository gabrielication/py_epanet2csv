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

df = pd.read_pickle("tensorflow_datasets/one_res_small/gabriele_marzo_2023/processed_df.pickle")

y = df.pop("has_leak")
X = df

shape_of_X = X.shape
len_of_X_cell = len(X.iloc[0]["4922"])
len_of_y_cell = len(y.iloc[0])

model = Sequential()
model.add(BatchNormalization(input_shape=(shape_of_X[0],shape_of_X[1], len_of_X_cell)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len_of_y_cell, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10)
