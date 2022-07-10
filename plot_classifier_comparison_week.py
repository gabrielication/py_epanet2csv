from sklearn.neighbors import KNeighborsClassifier

from sklearnex import patch_sklearn
patch_sklearn()

# We will patch just SVM since it is incredibly slower without intelex
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

names = [
	"KNeighborsClassifier",
	"LinearSVM",
	"RBFSVM",
	"DecisionTree",
	"RandomForest",
	"AdaBoostClassifier",
	"GaussianNaiveBayes",
	"MLPClassifier",
]


def execute_classifier_comparison_wo_smart_sensors(input_filename, prefix_output_filename):
	print("Loading " + input_filename + "...")

	data = pd.read_csv(input_filename,
					   names=["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
							  "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
							  "current_leak_demand_value", "smart_sensor_presence"])

	print("Rows number : ", len(data))
	print("Nodes number : ", len(data["nodeID"].unique()))

	# cut dataset to 168 hours
	end_index_dataset = len(data["nodeID"].unique())*24*7
	data = data.iloc[:end_index_dataset]

	print("Rows number : ", len(data))
	print("Nodes number : ", len(data["nodeID"].unique()))

	print("Dividing X and y matrices...\n")

	X = data[["demand_value", "head_value", "pressure_value"]].copy()
	y_sensors = pd.DataFrame(0, index=np.arange(len(data)), columns=["smart_sensor_presence"])
	y_sensors = y_sensors["smart_sensor_presence"].astype(int)
	y = data["has_leak"].astype(int)

	classifiers_configurator(X, y, y_sensors, prefix_output_filename, input_filename)


def execute_classifier_comparison_with_smart_sensors(input_filename, prefix_output_filename):
	print("WITH SENSORS. Loading " + input_filename + "...")

	data = pd.read_csv(input_filename,
					   names=["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
							  "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
							  "current_leak_demand_value", "smart_sensor_presence"])

	print("Rows number : ", len(data))
	print("Nodes number : ", len(data["nodeID"].unique()))

	# cut dataset to 168 hours
	end_index_dataset = len(data["nodeID"].unique()) * 24 * 7
	data = data.iloc[:end_index_dataset]

	print("Rows number : ", len(data))
	print("Nodes number : ", len(data["nodeID"].unique()))

	print("Dividing X and y matrices...\n")

	X = data[["demand_value", "head_value", "pressure_value", "smart_sensor_presence"]].copy()
	y_sensors = data["smart_sensor_presence"].astype(int)
	y = data["has_leak"].astype(int)

	classifiers_configurator(X, y, y_sensors, prefix_output_filename, input_filename)


def classifiers_configurator(X, y, y_sensors, prefix_output_filename, input_filename):
	output_filename = prefix_output_filename + 'prediction_accuracies_complete.csv'

	# open the file in the write mode
	f = open(output_filename, 'w+')

	# create the csv writer
	writer = csv.writer(f)

	header = ["Classificator", "Accuracy", "Precision", "Recall", "False Positive Rate"]
	writer.writerow(header)

	clf1 = Pipeline([('scaler', StandardScaler()), ('KNC', KNeighborsClassifier())])
	clf2 = Pipeline([('scaler', StandardScaler()), ('SVCL', SVC(kernel="linear"))])
	clf3 = Pipeline([('scaler', StandardScaler()), ('SVCR', SVC(kernel="rbf", gamma='auto'))])
	clf4 = Pipeline([('scaler', StandardScaler()), ('DTC', DecisionTreeClassifier())])
	clf5 = Pipeline([('scaler', StandardScaler()), ('RFC', RandomForestClassifier())])
	clf6 = Pipeline([('scaler', StandardScaler()), ('ABC', AdaBoostClassifier())])
	clf7 = Pipeline([('scaler', StandardScaler()), ('GNB', GaussianNB())])
	clf8 = Pipeline([('scaler', StandardScaler()), ('MLPC', MLPClassifier(random_state=1, max_iter=1000))])

	classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8]

	for name, clf in zip(names, classifiers):
		prediction_measurements = execute_classifier(clf, name, 7, X, y, y_sensors, prefix_output_filename)
		# print(name + "'s prediction accuracy (mean of kfolds) is: " + str(prediction_accuracy) + "\n")

		output_row = [name] + list(prediction_measurements)
		print(output_row)

		writer.writerow(output_row)

	# close the file
	f.close()

	print("\n" + input_filename + " comparison done!\n\n")


def execute_classifier(model, name, k_folds, X, y, y_sensors, prefix_output_filename):
	print("Executing " + name + "...")

	prediction_accuracy = 0.0

	conf_matrix_list_of_arrays = []
	# kf = KFold(n_splits=k_folds, random_state=1, shuffle=True)
	index_fold = 1
	kf = KFold(n_splits=k_folds, shuffle=False)
	for train_index, test_index in kf.split(X):
		print("k-fold n. : ", index_fold)
		print("start index train : ", train_index[0])
		print("start index test : ", test_index[0])
		# continue

		count_different_zero = 0
		count_different_one = 0

		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]

		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)

		# update y_pred with know sensor information, if present
		for ii in range(len(y_pred)):
			# no sensor
			if y_sensors.iloc[ii] == 0:
				continue
			# sensor not leak
			elif y_sensors.iloc[ii] == 1:
				if y_pred[ii] > 0:
					print("EXIT with different > 0 *************")
					count_different_one += 1
					# sys.exit(1)
				y_pred[ii] = 0
			# sensor and leak
			elif y_sensors.iloc[ii] == 2:
				if y_pred[ii] < 1:
					print("EXIT with different < 1 *************")
					count_different_zero += 1
					# sys.exit(1)
				y_pred[ii] = 1

		prediction_accuracy += accuracy_score(y_test, y_pred)

		print("count with different > 0 : ", count_different_one)
		print("count with different < 1 : ", count_different_one)

		conf_matrix = confusion_matrix(y_test, y_pred)
		conf_matrix_list_of_arrays.append(conf_matrix)

		index_fold += 1

	# sys.exit(1)
	prediction_accuracy = prediction_accuracy / k_folds

	# print(conf_matrix_list_of_arrays)

	mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)
	# print(mean_of_conf_matrix_arrays)

	group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

	group_counts = ["{0:0.0f}".format(value) for value in mean_of_conf_matrix_arrays.flatten()]
	print(group_counts)

	# sys.exit(1)

	true_negatives = float(group_counts[0])
	false_positives = float(group_counts[1])
	false_negatives = float(group_counts[2])
	true_positives = float(group_counts[3])

	calc_accuracy = (true_positives + true_negatives) / (
				true_positives + true_negatives + false_negatives + false_positives)

	precision = true_positives / (true_positives + false_positives)
	recall = true_positives / (true_positives + false_negatives)
	false_positive_rate = false_positives / (false_positives + true_negatives)

	group_percentages = ["{0:.2%}".format(value) for value in
						 mean_of_conf_matrix_arrays.flatten() / np.sum(mean_of_conf_matrix_arrays)]

	labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
			  zip(group_names, group_counts, group_percentages)]

	labels = np.asarray(labels).reshape(2, 2)

	ax = sns.heatmap(mean_of_conf_matrix_arrays, annot=labels, fmt='', cmap='Blues')

	ax.set_title(name + '\n\n');
	ax.set_xlabel('\nPredicted Values')
	ax.set_ylabel('Actual Values ');

	## Ticket labels - List must be in alphabetical order
	ax.xaxis.set_ticklabels(['False', 'True'])
	ax.yaxis.set_ticklabels(['False', 'True'])

	## Display the visualization of the Confusion Matrix.
	filename = prefix_output_filename + name + '.png'
	plt.savefig(filename)
	plt.clf()
	return calc_accuracy, precision, recall, false_positive_rate


if __name__ == "__main__":
	print("Plot classifier comparison started!\n")

	# input_filename = 'nodes_output.csv'

	# execute_classifier_comparison_wo_smart_sensors("1D_one_res_small/1D_one_res_small_nodes_output.csv",
	#                                                "1D_one_res_small/without_sensors/1D_one_res_small_")
	# execute_classifier_comparison_wo_smart_sensors("1D_one_res_large/1D_one_res_large_nodes_output.csv",
	#                                                "1D_one_res_large/without_sensors/1D_one_res_large_")
	# execute_classifier_comparison_wo_smart_sensors("1D_two_res_large/1D_two_res_large_nodes_output.csv",
	#                                                "1D_two_res_large/without_sensors/1D_two_res_large_")

	# execute_classifier_comparison_wo_smart_sensors("1M_one_res_small/1M_one_res_small_nodes_output.csv",
	#                                                "1M_one_res_small_4_fold_week/without_sensors/1M_one_res_small_")
	# execute_classifier_comparison_wo_smart_sensors("1M_one_res_large/1M_one_res_large_nodes_output.csv",
	#                                                "1M_one_res_large_4_fold_week/without_sensors/1M_one_res_large_")
	# execute_classifier_comparison_wo_smart_sensors("1M_two_res_large/1M_two_res_large_nodes_output.csv",
	#                                                "1M_two_res_large_4_fold_week/without_sensors/1M_two_res_large_")
	#
	# # Add sensor field
	# execute_classifier_comparison_with_smart_sensors("1D_one_res_small/1D_one_res_small_nodes_output.csv",
	#                                                  "1D_one_res_small/with_sensors/1D_one_res_small_")
	# execute_classifier_comparison_with_smart_sensors("1D_one_res_large/1D_one_res_large_nodes_output.csv",
	#                                                  "1D_one_res_large/with_sensors/1D_one_res_large_")
	# execute_classifier_comparison_with_smart_sensors("1D_two_res_large/1D_two_res_large_nodes_output.csv",
	#                                                  "1D_two_res_large/with_sensors/1D_two_res_large_")
	#
	#
	# execute_classifier_comparison_with_smart_sensors("1M_one_res_small/1M_one_res_small_nodes_output.csv",
	#                                                  "1M_one_res_small_4_fold_week/with_sensors/1M_one_res_small_")
	# execute_classifier_comparison_with_smart_sensors("1M_one_res_large/1M_one_res_large_nodes_output.csv",
	#                                                  "1M_one_res_large_4_fold_week/with_sensors/1M_one_res_large_")
	# execute_classifier_comparison_with_smart_sensors("1M_two_res_large/1M_two_res_large_nodes_output.csv",
	#                                                  "1M_two_res_large_4_fold_week/with_sensors/1M_two_res_large_")



	execute_classifier_comparison_wo_smart_sensors("1W_one_res_small/1W_one_res_small_nodes_output.csv",
	                                                "1W_one_res_small_7_fold_day/without_sensors/1W_one_res_small_")
	execute_classifier_comparison_wo_smart_sensors("1W_one_res_large/1W_one_res_large_nodes_output.csv",
	                                                "1W_one_res_large_7_fold_day/without_sensors/1W_one_res_large_")
	execute_classifier_comparison_wo_smart_sensors("1W_two_res_large/1W_two_res_large_nodes_output.csv",
	                                                "1W_two_res_large_7_fold_day/without_sensors/1W_two_res_large_")

	execute_classifier_comparison_with_smart_sensors("1W_one_res_small/1W_one_res_small_nodes_output.csv",
													 "1W_one_res_small_7_fold_day/with_sensors/1W_one_res_small_")
	execute_classifier_comparison_with_smart_sensors("1W_one_res_large/1W_one_res_large_nodes_output.csv",
													 "1W_one_res_large_7_fold_day/with_sensors/1W_one_res_large_")
	execute_classifier_comparison_with_smart_sensors("1W_two_res_large/1W_two_res_large_nodes_output.csv",
													 "1W_two_res_large_7_fold_day/with_sensors/1W_two_res_large_")
