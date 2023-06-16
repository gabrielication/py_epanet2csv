import numpy as np
import pandas as pd
import os
import shutil
import csv
import sys
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from tensorflow.keras import layers


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
# from keras_visualizer import visualizer

def clean_old_files():
	print("FRESH START ENABLED. Cleaning ALL old models and their files...")

	for filename in Path(".").glob("*.png"):
		try:
			os.remove(filename)
			print(str(filename) + " deleted")
		except OSError:
			print("\nError while deleting " + str(filename) + "\n")

	for filename in Path(".").glob("my_model"):
		try:
			shutil.rmtree(filename)

			print(str(filename) + " deleted")
		except OSError:
			print("\nError while deleting " + str(filename) + "\n")

	for filename in Path(".").glob("my_history.npy"):
		try:
			os.remove(filename)
			print(str(filename) + " deleted")
		except OSError:
			print("\nError while deleting " + str(filename) + "\n")

	print("All old files deleted.\n")

def fit_and_or_load_model(train_features, train_labels, epochs, validation_split, batch_size, callbacks, complete_path_stat, save_model=True, visualize_model_bool=True):

	input_filename_full_fitted_model = ""

	for filename in Path(".").glob("tensorflow_models/domenico_demand_prediction_model"):
		input_filename_full_fitted_model = str(filename)
		# cheap hack. we just have one model file. break at first finding.
		break

	if input_filename_full_fitted_model != "":
		print("Full fitted model already exists. Loading " + input_filename_full_fitted_model + "...")
		# model_fitted = load(input_filename_full_fitted_model)

		model = tf.keras.models.load_model(input_filename_full_fitted_model)

		history = np.load('tensorflow_models/domenico_demand_prediction_model/demand_prediction_model.npy', allow_pickle='TRUE').item()


		return model, history
	else:
		# we first fit the model on the complete dataset and save the fitted model back
		print("No old model found. Creating and Fitting...")

		model = create_regressor_nn_model(train_features, complete_path_stat, normalize=True)

		history = perform_neural_network_fit(model, train_features, train_labels, epochs,
											 validation_split=validation_split, batch_size=batch_size,
											 callbacks=callbacks, verbose=1)

		if(save_model):
			np.save('tensorflow_models/domenico_demand_prediction_model/demand_prediction_model.npy', history.history)

			output_filename_full_fitted_model = 'tensorflow_models/domenico_demand_prediction_model'
			model.save(output_filename_full_fitted_model)

			print("Model saved to: " + output_filename_full_fitted_model)
		else:
			print("Model and History NOT SAVED!")


		if(visualize_model_bool):
			visualize_model(model)
		else:
			print("Model NOT VISUALIZED!")

		return model, history
	# model = create_neural_network_model(train_features, complete_path_stat, normalize=True)
	#
	# history = perform_neural_network_fit(model, train_features, train_labels, epochs,
	# 									 validation_split=validation_split, batch_size=batch_size,
	# 									 callbacks=callbacks, verbose=1)
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

def load_dataset(complete_path, cols, scaling=False, pairplot=False):
	print("LOADING " + complete_path + "...")

	# We read our entire dataset
	data = pd.read_csv(complete_path,  delimiter=';')

	data = data.drop(data[data["nodeID"]=="7384"].index)

	if cols:
		# We drop these colum
		print("Extracting only columns: ", cols)
		data_trans = data[cols].copy()
	else:
		data_trans = data.copy()

	# # Convert the types of the desired columns and add them back
	# le = preprocessing.LabelEncoder()
	# data_trans["hour"] = le.fit_transform(data["hour"])
	# data_trans["nodeID"] = le.fit_transform(data["nodeID"])
	# data_trans["node_type"] = le.fit_transform(data["node_type"])
	# data_trans["has_leak"] = le.fit_transform(data["has_leak"])

	print(data_trans.columns)

	data_scaled = data_trans

	if(scaling):
		scaler = StandardScaler()

		print("Standard Scaling IS ACTIVE. Preprocessing...")
		scaler.fit(data_trans)
		data_scaled = scaler.transform(data_trans)
		data_scaled = pd.DataFrame(data_scaled, columns=[cols])

		print(data_trans.head())
		print(data_scaled.head())
		print("Preprocessing done.\n")
	else:
		print("Standard Scaling IS NOT ACTIVE.")

	print("Dividing FEATURES and LABELS...")

	# This was used in Tensorflow wiki but it's not the same as train test split. It will pick a SAMPLE jumping rows, not a clean SPLIT
	# train_dataset = data_scaled.sample(frac=0.8, random_state=0)


	if(pairplot):
		now = formatted_datetime()
		output_filename = "pairplot_"+now+".png"
		# base_demand;demand_value;head_value;pressure_value
		sns.pairplot(data_scaled[["pressure_value", "head_value", "base_demand", "demand_value"]], diag_kind='kde').savefig(output_filename)
		print(output_filename+" saved.")


	# df = pd.read_csv(complete_path_stat)
	n_nodes = 83 #int(df['number_of_nodes'].iloc[0])
	duration = 168 #int(df['time_spent_on_sim'].iloc[0])
	duration_percentage = int(0.5 * duration)


	# ###########
	# ########### USE NUMPY
	# ###########
	# train_dataset_size = duration_percentage
	#
	# train_dataset = data_scaled.iloc[:train_dataset_size, :]
	# test_dataset = data_scaled.drop(train_dataset.index)
	#
	# cols_1 = ["pressure_value", "base_demand"]
	# label_1 = ["demand_value"]
	#
	# features = data_scaled[cols_1].values
	# labels = data_scaled[label_1].values
	#
	# #!!!! IMPORTANT
	# features = features.reshape(-1,83,2)
	# labels = labels.reshape(-1, 83)
	#
	# train_features = features[:train_dataset_size]
	# test_features = features[train_dataset_size:]
	#
	# train_labels = labels[:train_dataset_size]
	# test_labels = labels[train_dataset_size:]



	##########
	########## USE DATAFRAME
	##########
	train_dataset_size = duration_percentage * n_nodes

	train_dataset = data_scaled.iloc[:train_dataset_size, :]
	test_dataset = data_scaled.drop(train_dataset.index)

	# Tensorflow guide (https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/regression.ipynb#scrollTo=2l7zFL_XWIRu)
	# says that the features are the columns that we want our network to train and labels is the value(s) to predict

	train_features = train_dataset.copy()
	test_features = test_dataset.copy()

	node_names = ["8614", "8600", "8610", "9402", "8598", "8608", "8620", "8616", "4922", "J106", "8618", "8604", "8596", "9410", "8612", "8602", "8606", "5656", "8622",
				  "8624", "8626", "8628", "8630", "8644", "8634", "8632", "8636", "8646", "8688", "8640", "8642", "8638", "8698", "8692", "8648", "8690", "8718",
				  "8702", "8700", "8694", "8738", "8696", "8740", "8720", "8706", "8704", "8686", "8708", "8660", "8656", "8664", "8662", "8654", "8716", "8650",
				  "8746", "8732", "8684", "8668", "8730", "8658", "8678", "8652", "8676", "8714", "8710", "8712", "8682", "8666", "8674", "8742", "8680", "8672",
				  "8792", "8722", "8726", "8724", "8744", "8736", "8728", "8670", "8734", "7384"]

	for jj in range(len(node_names)):
		if not jj % 10 == 0:
			train_features.drop(train_features[train_features["nodeID"] == node_names[jj]].index)
			test_features.drop(test_features[test_features["nodeID"] == node_names[jj]].index)

	train_features = train_features.drop(['hour', 'nodeID', 'base_demand', 'demand_value', 'head_value',
										  'pressure_value', 'x_pos', 'y_pos', 'node_type',
										  'leak_area_value', 'leak_discharge_value', 'leak_demand_value'], axis=1)
	test_features = test_dataset.drop(['hour', 'nodeID', 'base_demand', 'demand_value', 'head_value',
										  'pressure_value', 'x_pos', 'y_pos', 'node_type',
										  'leak_area_value', 'leak_discharge_value', 'leak_demand_value'], axis=1)


	# These instructions modificate also original dataframes
	train_labels = train_features.pop('has_leak')
	test_labels = test_features.pop('has_leak')

	return train_dataset, test_dataset, train_features, test_features, train_labels, test_labels

def create_regressor_nn_model(train_features, complete_path_stat, normalize=True):
	print("Building Neural Network Model...")

	if(normalize):#(normalize):
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

		# feat_shape = train_features.shape[1]
		# input_layer = layers.Input(shape=(feat_shape,))

		feat_shape = 83 #train_features.shape[1]
		input_layer = layers.Input(shape=(feat_shape,2))

	# These lines will just calculate the levels for the Deep Neural Net
	df = pd.read_csv(complete_path_stat)
	n_junc = int(df['number_of_junctions'].iloc[0])

	fst_level = n_junc * 2
	snd_level = n_junc * 2
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
	# metrics = ['mse', 'mae']

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

	# history = model.fit(
	# 	train_features,
	# 	train_labels,
	# 	epochs=epochs,
	# 	batch_size=batch_size,
	# 	validation_split=validation_split,
	# 	verbose=verbose,
	# )

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

def evaluate_regression_after_fit(model, test_features, test_labels):
	print("Evaluation started...")

	evl = model.evaluate(test_features, test_labels, verbose=0)

	loss = evl[0]
	mse = evl[1]
	mae = evl[2]
	r_square = None
	# r_square = evl[3]

	print("loss: ",loss)
	print("mse: ",mse)
	print("mae: ",mae)
	print("r_square: ",r_square)

	return loss, mse, mae, r_square

def evaluate_classification_after_fit(model, test_features, test_labels):
	print("Evaluation started...")

	evl = model.evaluate(test_features, test_labels, verbose=0)

	metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=1)]

	accuracy = evl[0]
	precision = evl[1]
	recall = evl[2]
	f1_score = None
	# f1_score = evl[3]

	print("accuracy: ",accuracy)
	print("precision: ",precision)
	print("recall: ",recall)
	print("f1_score: ",f1_score)

	return accuracy, precision, recall, f1_score

def predict_and_collect_results(model, test_features):
	print("Prediction started...")

	test_predictions = model.predict(test_features).flatten()

	return test_predictions

def create_classifier_nn_model(train_features):
	print("NORMALIZATION IS ENABLED!")
	# We want to Normalize (scale) the data since it can be too different in ranges
	# These lines will create a NORMALIZATION layer

	normalizer = tf.keras.layers.Normalization(axis=-1)
	normalizer.adapt(np.array(train_features))

	input_layer = normalizer

	# These lines will just calculate the levels for the Deep Neural Net
	df = pd.read_csv(complete_path_stat)
	n_junc = int(df['number_of_junctions'].iloc[0])

	fst_level = n_junc * 5
	snd_level = n_junc * 3
	trd_level = n_junc

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
	metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=1)]

	model.compile(loss=lossfn,
				  metrics=metrics,
				  optimizer=opt)

	# Can be skipped
	model.summary()

	return model


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

	evaluate_regression_after_fit(model, test_features, test_labels)

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

	loss, mse, mae, r_square = evaluate_regression_after_fit(model, test_features, test_labels)

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

	for pred,test in zip(test_predictions,test_labels.values):
		output_row = [pred,test,pred-test]
		writer.writerow(output_row)

	f.close()

	print("\nPrediction report saved to: "+output_filename)
	print("Quitting...")


def create_file_prediction_report(input_full_dataset, test_predictions, test_labels, test_has_leak, scenario, node_list, fresh_start=False):
	now = formatted_datetime()

	# output_filename = input_full_dataset[0:3] + "tensorflow_report_" + now + ".csv"
	output_filename = input_full_dataset[0:10] + "_" + scenario + "_test_prediction_report_" + now + ".csv"

	# open the file in the write mode
	f = open(output_filename, "w", newline='', encoding='utf-8')

	# create the csv writer
	writer = csv.writer(f)

	header = ["nodeID","predictions","true_test_values","error","leakage"]

	writer.writerow(header)

	if (fresh_start):
		clean_old_files()

	for nodeID,pred,test,has_leak in zip(node_list, test_predictions,test_labels.values,test_has_leak.values):
		output_row = [nodeID,pred,test,pred-test,has_leak]
		writer.writerow(output_row)

	f.close()

	print("\nPrediction report saved to: "+output_filename)
	print("Quitting...")


# if __name__ == "__main__":
# 	print('Tensorflow ', tf.__version__)
# 	print('Keras ', tf.keras.__version__)
# 	is_gpu_supported()
#
# 	folder_input = "tensorflow_datasets/"
#
# 	folder_network = "one_res_small/no_leaks_rand_base_demand/1W/"
# 	# input_full_dataset = folder_network + '1W_one_res_small_no_leaks_rand_base_dem_nodes_output.csv'
# 	# input_full_dataset = folder_network + '1W_one_res_small_no_leaks_rand_bd_merged.csv'
# 	#
# 	folder_network = "one_res_small/no_leaks_rand_base_demand/1M/"
# 	input_full_dataset = folder_network + '1M_one_res_small_no_leaks_rand_bd_merged.csv'
#
# 	input_stat_full_dataset = folder_network + "1W_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"
#
#
#
# 	input_alt_dataset = [folder_network + "1W_ALT_one_res_small_with_1_leaks_rand_base_dem_nodes_output.csv",
# 						 folder_network + "1W_ALT_one_res_small_with_1_at_8_leaks_rand_base_dem_nodes_output.csv",
# 						 folder_network + "1W_ALT_one_res_small_with_1_at_4_leaks_rand_base_dem_nodes_output.csv",
# 						 folder_network + "1W_ALT_one_res_small_with_1_at_2_leaks_rand_base_dem_nodes_output.csv"
# 						 ]
#
# 	epochs = 100
# 	batch_size = 10 #number of nodes
#
# 	cols = ["pressure_value", "base_demand"]
# 	label = "demand_value"
#
# 	# create_analysis_report(folder_input, input_full_dataset, input_alt_dataset, input_stat_full_dataset, cols, label,
# 	#                        epochs, fresh_start=True)
#
# 	create_prediction_report(folder_input, input_full_dataset, input_alt_dataset, input_stat_full_dataset, cols, label,
# 							 epochs, fresh_start=True, batch_size=batch_size)
#
# 	# folder_input = ""
#
# 	# input_full_dataset = '1W_Net3_no_leaks_nodes_output.csv'
# 	# input_stat_full_dataset = "1W_Net3_no_leaks_nodes_simulation_stats.csv"
# 	#
#
# 	# input_full_dataset = '1W_Net3_no_leaks_nodes_output.csv'
# 	# input_stat_full_dataset = "1W_Net3_no_leaks_nodes_simulation_stats.csv"
# 	#
# 	# input_alt_dataset = [
# 	#                      "1W_Net3_with_leaks_nodes_output.csv"
# 	#                      ]
# 	#
#
# 	#
# 	# print("FRESH START ENABLED. Cleaning ALL old models and their files...")
# 	# clean_old_files()
# 	#
# 	# complete_path = folder_input + input_full_dataset
# 	# complete_path_stat = folder_input + input_stat_full_dataset
# 	#
# 	# cols.append(label)
# 	#
# 	# run_evaluation_analysis(complete_path, complete_path_stat, epochs, cols)




if __name__ == "__main__":
	print('Tensorflow ', tf.__version__)
	print('Keras ', tf.keras.__version__)
	is_gpu_supported()

	folder_input = "tensorflow_group_datasets/"

	### 1M ###

	### 1M no leak
	folder_network = "one_res_small/no_leaks_rand_base_demand/"
	input_full_dataset = folder_network + '1M_one_res_small_leaks_ordered_group_0_node_0_0164_merged.csv'
	complete_path = folder_input + input_full_dataset

	### 1M leak
	folder_network_leakage = "one_res_small/1_at_4_leaks_rand_base_demand/"
	# input_full_dataset_leakage = folder_network_leakage + '1M_one_res_small_leaks_ordered_group_merged.csv'
	input_full_dataset_leakage = folder_network_leakage + '1M_one_res_small_leaks_ordered_group_0082_merged.csv'
	complete_path_leakage = folder_input + input_full_dataset_leakage


	### 1M leak
	folder_network_leakage_2 = "one_res_small/1_at_4_leaks_rand_base_demand/"
	input_full_dataset_leakage_2 = folder_network_leakage_2 + '1M_one_res_small_leaks_ordered_group_8_0082_merged.csv'
	complete_path_leakage_2 = folder_input + input_full_dataset_leakage_2

	# cols = ["nodeID", "pressure_value", "base_demand", "demand_value", "has_leak"]
	cols = None

	# load dati senza perdita
	train_dataset, test_dataset, train_features, test_features, train_labels, test_labels = load_dataset(complete_path,
																										 cols,
																										 scaling=False,
																										 pairplot=False)

	# load dati con perdita
	train_dataset_leakage, test_dataset_leakage, train_features_leakage, \
		test_features_leakage, train_labels_leakage, test_labels_leakage = load_dataset(complete_path_leakage,
																										 cols,
																										 scaling=False,
																										 pairplot=False)


	# load dati con perdita 2
	train_dataset_leakage_2, test_dataset_leakage_2, train_features_leakage_2, \
		test_features_leakage_2, train_labels_leakage_2, test_labels_leakage_2 = load_dataset(complete_path_leakage_2,
																										 cols,
																										 scaling=False,
																										 pairplot=False)



	epochs = 100
	batch_size = 10 #10

	print("PREDICT ANALYSIS:\n")

	validation_split = 0.2
	earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
	callbacks = [earlystop]





	train_features.pop("has_leak")
	train_features.pop("nodeID")

	# creazione modello basato su training dati senza perdita
	model, history = fit_and_or_load_model(train_features, train_labels, epochs, validation_split, batch_size,
										   callbacks, complete_path_stat, save_model=True, visualize_model_bool=False)



	#test del modello sui dati di train senza perdita
	# train_features.pop('has_leak')
	evaluate_regression_after_fit(model, train_features, train_labels)
	training_predictions = predict_and_collect_results(model, train_features)

	# mse = np.mean(np.abs(np.array(test_predictions) - np.array(test_labels)) ** 2, axis=0)
	# mae = np.mean(np.abs(np.array(test_predictions) - np.array(test_labels)), axis=0)
	# print(mse)
	# print(mae)



	print("********************************")
	print("REPORT TEST DATASET (no leak)")
	print("********************************")
	#test del modello sui dati di test senza perdita
	test_has_leak = test_features['has_leak']
	test_features.pop('has_leak')
	test_features.pop("nodeID")

	evaluate_regression_after_fit(model, test_features, test_labels)
	test_predictions = predict_and_collect_results(model, test_features)

	# mse = np.mean(np.abs(np.array(test_predictions) - np.array(test_labels)) ** 2, axis=0)
	# mae = np.mean(np.abs(np.array(test_predictions) - np.array(test_labels)), axis=0)
	# print(mse)
	# print(mae)

	create_file_prediction_report(input_full_dataset, test_predictions, test_labels, test_has_leak, "NO_LEAK", test_dataset.nodeID.values, fresh_start=True)



	print("********************************")
	print("REPORT LEAK TEST DATASET (leak)")
	print("********************************")
	#test del modello sui dati di test con perdita
	test_has_leak_leakage = test_features_leakage['has_leak']
	test_features_leakage.pop('has_leak')
	test_features_leakage.pop('nodeID')
	evaluate_regression_after_fit(model, test_features_leakage, test_labels_leakage)
	test_predictions_leakage = predict_and_collect_results(model, test_features_leakage)

	# mse = np.mean(np.abs(np.array(test_predictions_leakage) - np.array(test_labels_leakage)) ** 2, axis=0)
	# mae = np.mean(np.abs(np.array(test_predictions_leakage) - np.array(test_labels_leakage)), axis=0)
	# print(mse)
	# print(mae)

	create_file_prediction_report(input_full_dataset_leakage, test_predictions_leakage, test_labels_leakage, test_has_leak_leakage, "LEAK", test_dataset_leakage.nodeID.values, fresh_start=True)

	sys.exit(1)


	#test del modello sui dati di traning con perdita
	train_features_leakage.pop('has_leak')
	evaluate_regression_after_fit(model, train_features_leakage, train_labels_leakage)
	train_predictions_leakage = predict_and_collect_results(model, train_features_leakage)

	# mse = np.mean(np.abs(np.array(train_predictions_leakage) - np.array(train_labels_leakage)) ** 2, axis=0)
	# mae = np.mean(np.abs(np.array(train_predictions_leakage) - np.array(train_labels_leakage)), axis=0)
	# print(mse)
	# print(mae)


	sys.exit(1)

	#######################################
	#creazione features per classificazione --> demand_value vera e predetta
	#######################################
	#matrice di due colonne --> demand_valuie vera e demand_value predetta per il dataset senza perdita
	classifier_train_features_no_leakage = np.array([test_labels, test_predictions]) # qui ci sono i demand_value veri e predetti
	classifier_train_features_no_leakage = classifier_train_features_no_leakage.T

	# matrice di due colonne --> demand_valuie vera e demand_value predetta per il dataset con perdita
	classifier_train_features_leakage = np.array([test_labels_leakage, test_predictions_leakage])
	classifier_train_features_leakage = classifier_train_features_leakage.T

	# concateno le due matrici di prima una sotto l'altra --> il risultato è la matrice di features che devo usare per la classificazione
	classifier_train_features = np.concatenate((classifier_train_features_no_leakage, classifier_train_features_leakage))

	####################################
	#creazione label per classificazione --> has_leak
	####################################

	classifier_train_labels = np.concatenate((test_dataset["has_leak"].values, test_dataset_leakage["has_leak"].values))

	# Creazione del modello classificatore
	# TODO: implementare il save/load del modello
	nn_model = create_classifier_nn_model(classifier_train_features)

	history = perform_neural_network_fit(nn_model, classifier_train_features, classifier_train_labels, epochs,
							   batch_size=batch_size, validation_split=validation_split,callbacks=callbacks, verbose=1)

	evaluate_classification_after_fit(nn_model, classifier_train_features, classifier_train_labels)



	#######################################
	#creazione features per classificazione --> demand_value vera e predetta
	#######################################
	#matrice di due colonne --> demand_valuie vera e demand_value predetta per il dataset senza perdita
	classifier_test_features_no_leakage = np.array([train_labels, training_predictions]) # qui ci sono i demand_value veri e predetti
	classifier_test_features_no_leakage = classifier_test_features_no_leakage.T

	# matrice di due colonne --> demand_valuie vera e demand_value predetta per il dataset con perdita
	classifier_test_features_leakage = np.array([train_labels_leakage, train_predictions_leakage])
	classifier_test_features_leakage = classifier_test_features_leakage.T

	# concateno le due matrici di prima una sotto l'altra --> il risultato è la matrice di features che devo usare per la classificazione
	classifier_test_features = np.concatenate((classifier_test_features_no_leakage, classifier_test_features_leakage))

	####################################
	#creazione label per classificazione --> has_leak
	####################################

	classifier_test_labels = np.concatenate((train_dataset["has_leak"].values, train_dataset_leakage["has_leak"].values))


	# TODO: what tests should we use?
	evaluate_classification_after_fit(nn_model, classifier_test_features, classifier_test_labels)

	test_classifier_predictions = nn_model.predict(classifier_test_features).flatten()

	now = formatted_datetime()

	# output_filename = input_full_dataset[0:3] + "tensorflow_report_" + now + ".csv"
	output_filename = "test_classificaiton_report_" + now + ".csv"

	# open the file in the write mode
	f = open(output_filename, "w", newline='', encoding='utf-8')

	# create the csv writer
	writer = csv.writer(f)

	header = ["predictions","true_test_values","error"]

	writer.writerow(header)

	for pred, test in zip(test_classifier_predictions, classifier_test_labels.values):
		output_row = [pred, test, pred - test]
		writer.writerow(output_row)

	f.close()

	# plot_predictions(test_predictions, test_labels)
