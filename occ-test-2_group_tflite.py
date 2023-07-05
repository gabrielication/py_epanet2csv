import sys

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




#### MAIN SCRIPT


#%%
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# %%
METRICS = [
	keras.metrics.TruePositives(name='tp'),
	keras.metrics.FalsePositives(name='fp'),
	keras.metrics.TrueNegatives(name='tn'),
	keras.metrics.FalseNegatives(name='fn'),
	keras.metrics.BinaryAccuracy(name='accuracy'),
	keras.metrics.Precision(name='precision'),
	keras.metrics.Recall(name='recall'),
	keras.metrics.AUC(name='auc'),
	keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def make_model(metrics=METRICS, output_bias=None):
	if output_bias is not None:
		output_bias = tf.keras.initializers.Constant(output_bias)
	model = keras.Sequential([
		keras.layers.Dense(
			38*2, activation='relu',
			input_shape=(train_features.shape[-1],)),
		keras.layers.Dropout(0.5),
		keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
	])

	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=1e-3),
		loss=keras.losses.BinaryCrossentropy(),
		metrics=metrics)

	return model


def plot_loss(history, label, n):
	# Use a log scale on y-axis to show the wide range of values.
	plt.semilogy(history.epoch, history.history['loss'],
			   color=colors[n], label='Train ' + label)
	plt.semilogy(history.epoch, history.history['val_loss'],
			   color=colors[n], label='Val ' + label,
			   linestyle="--")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()


def plot_metrics(history):
	metrics = ['loss', 'prc', 'precision', 'recall']
	for n, metric in enumerate(metrics):
		name = metric.replace("_", " ").capitalize()
		plt.subplot(2, 2, n + 1)
		plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
		plt.plot(history.epoch, history.history['val_' + metric],
				 color=colors[0], linestyle="--", label='Val')
		plt.xlabel('Epoch')
		plt.ylabel(name)
		if metric == 'loss':
			plt.ylim([0, plt.ylim()[1]])
		elif metric == 'auc':
			plt.ylim([0.8, 1.1])
		else:
			plt.ylim([0, 1.1])

		plt.legend();

def plot_cm(labels, predictions, title, p=0.5):
	cm = confusion_matrix(labels, predictions > p)
	plt.figure(figsize=(5, 5))
	sns.heatmap(cm, annot=True, fmt="d")
	# plt.title('Confusion matrix @{:.2f}'.format(p))
	plt.title(title)
	plt.ylabel('Actual label')
	plt.xlabel('Predicted label')

	print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
	print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
	print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
	print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
	print('Total Fraudulent Transactions: ', np.sum(cm[1]))

#%%
file = tf.keras.utils
# raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
# raw_df.head()

exported_path = 'tensorflow_group_datasets/one_res_small/0_no_leaks_rand_base_demand/'
dataset_path = exported_path

out_filename = '1M_one_res_small_leaks_ordered_group_0_node_0_0164_merged.csv'
raw_df1 = pd.read_csv(dataset_path+out_filename, delimiter=";")

exported_path = 'tensorflow_group_datasets/one_res_small/1_at_82_leaks_rand_base_demand/'
dataset_path = exported_path
# leak_node = 3

leak_area = "0164" # "0246" #"0164"
leak_group = 7

# %%
import csv

out_filename_complete = "report_model_comparison_leak_group_" + str(leak_group)  + ".csv"
out = open(out_filename_complete, "w", newline='', encoding='utf-8')
writer = csv.writer(out, delimiter=';')
header = ["leak_group_model", "leak_node_model", "leak_node_test", "loss", "accuracy"]
writer.writerow(header)


for leak_node_ii in range(1,10,1):
	leak_node = leak_node_ii
	out_filename = "1M_one_res_small_leaks_ordered_group_"+str(leak_group)+"_node_"+str(leak_node)+"_"+leak_area+"_merged.csv"
	raw_df2 = pd.read_csv(dataset_path+out_filename, delimiter=";")


	# Appending multiple DataFrame
	raw_df = pd.concat([raw_df1, raw_df2])#raw_df4, raw_df5, raw_df6, raw_df7, raw_df8])
	# options = ["8626", "8628", "8630", "8644", "8634", "8632", "8636", "8646", "8688", "8640"]
	# raw_df = raw_df.loc[raw_df['nodeID'].isin(options)]

	raw_df.reset_index(drop=True, inplace=True)

	# leak_area = "0164" # "0246" #"0164"
	# leak_group_test = leak_group
	# leak_node_test = 5
	# out_filename = "1M_one_res_small_leaks_ordered_group_"+str(leak_group_test)+"_node_"+str(leak_node_test)+"_"+leak_area+"_merged.csv"
	# raw_df_test = pd.read_csv(dataset_path+out_filename, delimiter=";")

	cleaned_df = raw_df.copy()
	# cleaned_df_test = raw_df_test.copy()

	#hour', 'nodeID', 'base_demand', 'demand_value', 'head_value',
		   # 'pressure_value', 'x_pos', 'y_pos', 'node_type', 'has_leak',
		   # 'leak_area_value', 'leak_discharge_value', 'leak_demand_value',
		   # 'flow_demand_in', 'demand_0', 'head_0', 'pressure_0', 'demand_1',
		   # 'head_1', 'pressure_1', 'demand_2', 'head_2', 'pressure_2', 'demand_3',
		   # 'head_3', 'pressure_3', 'demand_4', 'head_4', 'pressure_4', 'demand_5',
		   # 'head_5', 'pressure_5', 'demand_6', 'head_6', 'pressure_6', 'demand_7',
		   # 'head_7', 'pressure_7', 'demand_8', 'head_8', 'pressure_8', 'demand_9',
		   # 'head_9', 'pressure_9', 'flow_demand_out', 'leak_group'

	pop_col = ['hour', 'nodeID', 'node_type', 'leak_area_value', 'leak_discharge_value', 'leak_demand_value',
			   # 'leak_group',
			   'has_leak'
			   ]

	cleaned_df = cleaned_df.drop(pop_col, axis=1)
	# cleaned_df_test = cleaned_df_test.drop(pop_col, axis=1)

	cleaned_df.rename(columns = {'leak_group':'Class'}, inplace = True)
	# cleaned_df.rename(columns = {'has_leak':'Class'}, inplace = True)

	# cleaned_df_test.rename(columns = {'leak_group':'Class'}, inplace = True)
	# # cleaned_df_test.rename(columns = {'has_leak':'Class'}, inplace = True)

	#%%
	neg, pos = np.bincount(cleaned_df['Class'])
	total = neg + pos
	print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
		total, pos, 100 * pos / total))


	#%%
	# Use a utility from sklearn to split and shuffle your dataset.
	train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
	train_df, val_df = train_test_split(train_df, test_size=0.2)

	# Form np arrays of labels and features.
	train_labels = np.array(train_df.pop('Class'))
	bool_train_labels = train_labels != 0
	val_labels = np.array(val_df.pop('Class'))
	test_labels = np.array(test_df.pop('Class'))

	train_features = np.array(train_df)
	val_features = np.array(val_df)
	test_features = np.array(test_df)

	# test2_labels = np.array(cleaned_df_test.pop('Class'))
	# test2_features = np.array(cleaned_df_test)

	#%%
	scaler = StandardScaler()
	train_features = scaler.fit_transform(train_features)

	val_features = scaler.transform(val_features)
	test_features = scaler.transform(test_features)
	# test2_features = scaler.transform(test2_features)

	# train_features = np.clip(train_features, -5, 5)
	# val_features = np.clip(val_features, -5, 5)
	# test2_features = np.clip(test2_features, -5, 5)
	# test_features = np.clip(test_features, -5, 5)

	#%%
	pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
	neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)


	#%%
	EPOCHS = 600
	BATCH_SIZE = 2048

	early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor='val_prc',
		verbose=1,
		patience=30,
		mode='max',
		restore_best_weights=True)

	#%%
	model = make_model()
	model.summary()

	# #%%
	# model.predict(train_features[:10])
	# #%%
	# results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
	# print("Loss: {:0.4f}".format(results[0]))

	#%%
	initial_bias = np.log([pos/neg])
	initial_bias
	#%%
	model = make_model(output_bias=initial_bias)
	model.predict(train_features[:10])

	#%%
	results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
	print("Loss: {:0.4f}".format(results[0]))

	#%%
	initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
	model.save_weights(initial_weights)

	# #%%
	# model = make_model()
	# model.load_weights(initial_weights)
	# model.layers[-1].bias.assign([0.0])
	# zero_bias_history = model.fit(
	# 	train_features,
	# 	train_labels,
	# 	batch_size=BATCH_SIZE,
	# 	epochs=20,
	# 	validation_data=(val_features, val_labels),
	# 	verbose=0)
	#
	# #%%
	# model = make_model()
	# model.load_weights(initial_weights)
	# careful_bias_history = model.fit(
	# 	train_features,
	# 	train_labels,
	# 	batch_size=BATCH_SIZE,
	# 	epochs=20,
	# 	validation_data=(val_features, val_labels),
	# 	verbose=0)
	#
	# #%%


	#%%
	# plot_loss(zero_bias_history, "Zero Bias", 0)
	# plot_loss(careful_bias_history, "Careful Bias", 1)

	#%%
	model = make_model()
	model.load_weights(initial_weights)
	baseline_history = model.fit(
		train_features,
		train_labels,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		callbacks=[early_stopping],
		validation_data=(val_features, val_labels))

	#%%
	# plot_metrics(baseline_history)

	#%%
	train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)

	test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

	#%%
	baseline_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
	for name, value in zip(model.metrics_names, baseline_results):
		print(name, ': ', value)
	print()

	# plot_cm(test_labels, test_predictions_baseline, '')

	#%%
	# test2_predictions_baseline = model.predict(test2_features, batch_size=BATCH_SIZE)

	# baseline_results = model.evaluate(test2_features, test2_labels, batch_size=BATCH_SIZE, verbose=0)
	# for name, value in zip(model.metrics_names, baseline_results):
	# 	print(name, ': ', value)
	# print()
	#
	# plot_cm(test2_labels, test2_predictions_baseline, '')

	#%%
	# def plot_roc(name, labels, predictions, **kwargs):
	#   fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
	#
	#   plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
	#   plt.xlabel('False positives [%]')
	#   plt.ylabel('True positives [%]')
	#   plt.xlim([-0.5,20])
	#   plt.ylim([80,100.5])
	#   plt.grid(True)
	#   ax = plt.gca()
	#   ax.set_aspect('equal')

	#%%
	# plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
	# plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
	# plt.legend(loc='lower right');

	# model.save('tensorflow_group_datasets/model/h5/model_leak_group' + str(leak_group) + "_train_node_" + str(leak_node) + '.h5')
	save_model = True
	if(save_model):
		np.save('tensorflow_group_datasets/model/exported/model_leak_group' + str(leak_group) + "_train_node_" + str(leak_node) + '.npy', baseline_history.history)

		mobilenet_save_path = 'tensorflow_group_datasets/model/exported/model_leak_group' + str(leak_group) + "_train_node_" + str(leak_node)
		# model.save(output_filename_full_fitted_model)
		tf.saved_model.save(model, mobilenet_save_path)

		print("Model saved to: " + mobilenet_save_path)

		# loaded = tf.saved_model.load(mobilenet_save_path)
		# Convert the model
		converter = tf.lite.TFLiteConverter.from_saved_model(mobilenet_save_path)  # path to the SavedModel directory
		tflite_model = converter.convert()

		# Save the model.
		mobilenet_save_path_tflite = 'tensorflow_group_datasets/model/tflite/model_leak_group' + str(leak_group) + "_train_node_" + str(leak_node) + '.tflite'
		with open(mobilenet_save_path_tflite, 'wb') as f:
			f.write(tflite_model)
	else:
		print("Model and History NOT SAVED!")

	sys.exit(1)

	for final_test_index in range(1,10,1):
		leak_node_final_test = final_test_index
		out_filename = "1M_one_res_small_leaks_ordered_group_"+str(leak_group)+"_node_"+str(leak_node_final_test)+"_"+leak_area+"_merged.csv"
		print(out_filename)
		raw_df_test_2 = pd.read_csv(dataset_path+out_filename, delimiter=";")

		cleaned_df_test_2 = raw_df_test_2.copy()
		cleaned_df_test_2 = cleaned_df_test_2.drop(pop_col, axis=1)
		cleaned_df_test_2.rename(columns = {'leak_group':'Class'}, inplace = True)
		# cleaned_df_validation.rename(columns = {'has_leak':'Class'}, inplace = True)

		test2_labels_2 = np.array(cleaned_df_test_2.pop('Class'))
		test2_features_2 = np.array(cleaned_df_test_2)

		test2_features_2 = scaler.transform(test2_features_2)
		test2_predictions_baseline = model.predict(test2_features_2, batch_size=BATCH_SIZE)

		# baseline_results =model.evaluate(test2_features_2, test2_labels_2, batch_size=BATCH_SIZE, verbose=0)
		# for name, value in zip(model.metrics_names, baseline_results):
		#   print(name, ': ', value)
		# print()

		baseline_results =model.evaluate(test2_features_2, test2_labels_2, batch_size=BATCH_SIZE, verbose=0)
		for name, value in zip(model.metrics_names, baseline_results):
			if name == "loss":
				loss = value
				print(name, ': ', value)
			if name == "accuracy":
				acc = value
				print(name, ': ', value)
		print()

		# title = "group : "+ str(leak_group_final_test) + " - node : " + str(leak_node_final_test)
		# plot_cm(test2_labels_2, test2_predictions_baseline, title)

		out_row = [leak_group,leak_node,leak_node_final_test, loss, acc]
		writer.writerow(out_row)


out.close()
