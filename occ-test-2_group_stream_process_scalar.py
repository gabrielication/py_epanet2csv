"""
AUTHOR: Domenico Garlisi

USAGE:
	shall 0 (NOTIFY)
		python .\occ-test-2_group_stream_process_scalar.py --notify notify --verbose 1

	shall 1 (SERVER)
		cd D:\work\ECOBLU_smart_water\repository
		python .\occ-test-2_group_stream_process_scalar.py --server server --verbose 1 --logFileName report_compute_latency --serverNotifyIp

	shall 2
		cd D:\work\ECOBLU_smart_water\repository
		python .\occ-test-2_group_stream_process_scalar.py --client gateway_1 --gateway 1 --mqttBroker 127.0.0.1 --serverIp 127.0.0.1 --verbose 1

	shell 3
		python .\occ-test-2_group_stream_process_scalar.py --client gateway_2 --gateway 2 --mqttBroker 127.0.0.1 --serverIp 127.0.0.1 --verbose 1

	shall 4 (injection)
		cd /mnt/d/work/ELEGANT/repository/smart_metering_use_case/smart_meter/live_stats
		cargo r water

"""



import json
import sys
import time
import datetime

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns

# import sklearn
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# import time
import csv

from multiprocessing import Process, Lock
import zmq
import signal
import argparse
import paho.mqtt.client as mqtt

lock = Lock()

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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

leak_area = "0164" # "0246" #"0164"
# 1-->7 2-->7 3-->6 4-->2 5-->1 6-->7 7-->7
leak_group = 7
leak_group_model = 7

EPOCHS = 100
BATCH_SIZE = 2048

dataset_path = "tensorflow_group_datasets/model/h5/"
model_filename = "model_leak_group"+str(leak_group)+"_train_node_"+str(leak_group_model)+".h5"
loaded_model = tf.keras.models.load_model(dataset_path+model_filename)

# # Convert the model.
# converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
# tflite_model = converter.convert()

# # Save the model.
# dataset_tflite_path = "tensorflow_group_datasets/model/tflite/"
# model_tflite_filename = "model_leak_group"+str(leak_group)+"_train_node_"+str(leak_group_model)+".tflite"
# with open(dataset_tflite_path+model_tflite_filename, 'wb') as f:
#   f.write(tflite_model)


# dataset_path = 'tensorflow_group_datasets/one_res_small/1_at_82_leaks_rand_base_demand/'
dataset_path = 'tensorflow_group_datasets/one_res_small/1_at_82_leaks_rand_base_demand_scalar/'

parser = argparse.ArgumentParser(description='zeromq server/client')
parser.add_argument('--client')
parser.add_argument('--gateway')
parser.add_argument('--mqttBroker')
parser.add_argument('--serverIp')
parser.add_argument('--serverNotifyIp')

parser.add_argument('--server')
parser.add_argument('--logFileName')
parser.add_argument('--notify')

parser.add_argument('--verbose')

args = parser.parse_args()

p_threshold = 0.3



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

		plt.legend()

# setting callbacks for different events to see if it works, print the message etc.
def on_connect(client, userdata, flags, rc, properties=None):
	"""
		Prints the result of the connection with a reasoncode to stdout ( used as callback for connect )

		:param client: the client itself
		:param userdata: userdata is set when initiating the client, here it is userdata=None
		:param flags: these are response flags sent by the broker
		:param rc: stands for reasonCode, which is a code for the connection result
		:param properties: can be used in MQTTv5, but is optional
	"""
	print("CONNACK received with code %s." % rc)


# with this callback you can see if your publish was successful
def on_publish(client, userdata, mid, properties=None):
	"""
		Prints mid to stdout to reassure a successful publish ( used as callback for publish )

		:param client: the client itself
		:param userdata: userdata is set when initiating the client, here it is userdata=None
		:param mid: variable returned from the corresponding publish() call, to allow outgoing messages to be tracked
		:param properties: can be used in MQTTv5, but is optional
	"""
	print("mid: " + str(mid))


# print which topic was subscribed to
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
	"""
		Prints a reassurance for successfully subscribing

		:param client: the client itself
		:param userdata: userdata is set when initiating the client, here it is userdata=None
		:param mid: variable returned from the corresponding publish() call, to allow outgoing messages to be tracked
		:param granted_qos: this is the qos that you declare when subscribing, use the same one for publishing
		:param properties: can be used in MQTTv5, but is optional
	"""
	print("Subscribed: " + str(mid) + " " + str(granted_qos))






# def plot_cm(labels, predictions, title, p=0.5):
# 	cm = confusion_matrix(labels, predictions > p)
# 	plt.figure(figsize=(5, 5))
# 	sns.heatmap(cm, annot=True, fmt="d")
# 	# plt.title('Confusion matrix @{:.2f}'.format(p))
# 	plt.title(title)
# 	plt.ylabel('Actual label')
# 	plt.xlabel('Predicted label')
#
# 	print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
# 	print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
# 	print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
# 	print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
# 	print('Total Fraudulent Transactions: ', np.sum(cm[1]))

class CLIENT_Process:
# class CLIENT_Process(Process):

	def __init__(self, name, wn, gateway_num, mqttBroker, server_ip, verbose):
		Process.__init__(self)
		self.name = name
		self.wn = wn
		self.selected_groups_list = 0
		self.selected_gateway = gateway_num
		self.mqttBroker = mqttBroker
		self.server_ip = server_ip
		self.verbose = verbose
		self.socket = None
		self.groups = [['8614', '8600', '8610', '9402', '8598', '8608', '8620', '8616', '4922', 'J106'],
						['8618', '8604', '8596', '9410', '8612', '8602', '8606', '5656', '8622', '8624'],
						['8626', '8628', '8630', '8644', '8634', '8632', '8636', '8646', '8688', '8640'],
						['8642', '8638', '8698', '8692', '8648', '8690', '8718', '8702', '8700', '8694'],
						['8738', '8696', '8740', '8720', '8706', '8704', '8686', '8708', '8660', '8656'],
						['8664', '8662', '8654', '8716', '8650', '8746', '8732', '8684', '8668', '8730'],
						['8658', '8678', '8652', '8676', '8714', '8710', '8712', '8682', '8666', '8674'],
						 ['8742', '8680', '8672', '8792', '8722']]


	# print message, useful for checking if it was successful
	def on_message(self, client, userdata, msg):
		"""
			Prints a mqtt message to stdout ( used as callback for subscribe )
	
			:param client: the client itself
			:param userdata: userdata is set when initiating the client, here it is userdata=None
			:param msg: the message with topic and payload
		"""
		if self.verbose:
			print(msg.topic + " ")
		# print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
		json_payload = json.loads(msg.payload)


		"""
		/ELEGANT/lora/worker/1 1 b'{"hour":"4:00:00","nodeID":8606,"node_type":0,"has_leak": false,"leak_area_value":0.00000000,"leak_discharge_value":0.00000000,"leak_demand_value":0.00000000,"leak_group": false,"base
_demand_scalar":1.61402260,"demand_value_scalar":3.07904130,"head_value_scalar":1.83131029,"pressure_value_scalar":1.80781817,"x_pos_scalar":-1.70369627,"y_pos_scalar":0.71062079,"flow_demand_in_scalar":1.26283
951,"demand_0_scalar":2.58557606,"head_0_scalar":1.23454060,"pressure_0_scalar":1.23743265,"demand_1_scalar":1.81078698,"head_1_scalar":1.68737500,"pressure_1_scalar":1.68484377,"demand_2_scalar":0.52016348,"he
ad_2_scalar":1.79759973,"pressure_2_scalar":1.79248428,"demand_3_scalar":1.35266475,"head_3_scalar":1.72503764,"pressure_3_scalar":1.73009756,"demand_4_scalar":0.31099098,"head_4_scalar":1.62833329,"pressure_4_
scalar":1.66379797,"demand_5_scalar":2.52099018,"head_5_scalar":1.62117109,"pressure_5_scalar":1.62294726,"demand_6_scalar":3.06774545,"head_6_scalar":2.17445401,"pressure_6_scalar":2.18528809,"demand_7_scalar"
:0.79458963,"head_7_scalar":1.95281030,"pressure_7_scalar":2.00015975,"demand_8_scalar":0.29727010,"head_8_scalar":0.67698828,"pressure_8_scalar":0.60953755,"demand_9_scalar":-1.08162455,"head_9_scalar":0.35424
264,"pressure_9_scalar":0.28941521,"flow_demand_out_scalar":1.03708950,"tmst":1692020767400000,"dateTime":"2023-08-14 13:46:07.400939500 UTC","dev_addr":8606,"dev_eui":8606,"gateway":1}'
		"""

		input_data = [
			json_payload["base_demand_scalar"],json_payload["demand_value_scalar"], json_payload["head_value_scalar"], json_payload["pressure_value_scalar"],
			json_payload["x_pos_scalar"], json_payload["y_pos_scalar"], json_payload["flow_demand_in_scalar"],
			json_payload["demand_0_scalar"], json_payload["head_0_scalar"], json_payload["pressure_0_scalar"], json_payload["demand_1_scalar"], json_payload["head_1_scalar"], json_payload["pressure_1_scalar"],
			json_payload["demand_2_scalar"], json_payload["head_2_scalar"], json_payload["pressure_2_scalar"], json_payload["demand_3_scalar"], json_payload["head_3_scalar"], json_payload["pressure_3_scalar"],
			json_payload["demand_4_scalar"], json_payload["head_4_scalar"], json_payload["pressure_4_scalar"], json_payload["demand_5_scalar"], json_payload["head_5_scalar"], json_payload["pressure_5_scalar"],
			json_payload["demand_6_scalar"], json_payload["head_6_scalar"], json_payload["pressure_6_scalar"], json_payload["demand_7_scalar"], json_payload["head_7_scalar"], json_payload["pressure_7_scalar"],
			json_payload["demand_8_scalar"], json_payload["head_8_scalar"], json_payload["pressure_8_scalar"], json_payload["demand_9_scalar"], json_payload["head_9_scalar"], json_payload["pressure_9_scalar"],
			json_payload["flow_demand_out_scalar"]
		]
		if self.verbose:
			print("send data for : ", json_payload["nodeID"])
		# print(input_data[ii:ii + 1])
		now = datetime.datetime.now()
		# print(now.strftime("%d/%m/%Y %H:%M:%S"))
		# print(datetime.datetime.timestamp(now))

		json_str = json.dumps({
			"client": self.name,
			"nodeID": json_payload["nodeID"],
			"hour": json_payload["hour"],
			"tmst": json_payload["tmst"],
			"has_leak": str(json_payload["has_leak"]),
			"leak_group": str(json_payload["leak_group"]),
			"date": now.strftime("%d/%m/%Y %H:%M:%S"),
			"timestamp": datetime.datetime.timestamp(now),
			"data": input_data
		})

		# SEND
		# socket.send(input_data[ii:ii + 1])
		self.socket.send_json(json_str)

		# time.sleep(0.1)


	def run(self):
		print("######### START CLIENT ##########")
		print ("CLIENT NAME " + self.name)
		print("######### ############ ##########")
		# # Lock on
		# lock.acquire()
		# # Free lock
		# lock.release()

		# client --> connect to server
		context = zmq.Context()
		# socket = context.socket(zmq.REQ)
		self.socket = context.socket(zmq.PUSH)
		self.socket.connect('tcp://'+ self.server_ip +':5557')

		# client --> connect to MQTT broker
		port = 1885

		client = mqtt.Client(self.name)

		# setting callbacks, use separate functions like above for better visibility
		client.on_subscribe = on_subscribe
		client.on_message = self.on_message

		client.connect(self.mqttBroker, port)
		mqttTopic = "/ELEGANT/lora/worker/"+str(self.selected_gateway)
		# subscribe to all topics of encyclopedia by using the wildcard "#"
		print("subscribing to : ", mqttTopic)
		client.subscribe(mqttTopic, qos=1)


		# SEND
		# socket.send(args.client.encode("utf8"))
		# RECEIVE
		# msg = socket.recv()
		# print(msg)

		# for leak_node in range(7, 8, 1):
		# 	print("leak node : ", leak_node)
		# 	out_filename = "1M_one_res_small_leaks_ordered_group_" + str(leak_group) + "_node_" + str(leak_node) + "_" + leak_area + "_merged_scalar.csv"
		# 	raw_df = pd.read_csv(dataset_path + out_filename, delimiter=";")
		#
		# 	cleaned_df = raw_df.copy()
		# 	pop_col = ['hour', 'nodeID', 'node_type', 'leak_area_value', 'leak_discharge_value', 'leak_demand_value', 'has_leak']
		# 	cleaned_df = cleaned_df.drop(pop_col, axis=1)
		# 	cleaned_df.rename(columns={'leak_group': 'Class'}, inplace=True)
		#
		# 	# #%%
		# 	# neg, pos = np.bincount(cleaned_df['Class'])
		# 	# total = neg + pos
		# 	# print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
		# 	# 	total, pos, 100 * pos / total))
		#
		# 	# %%
		# 	# Use a utility from sklearn to split and shuffle your dataset.
		# 	# train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
		# 	# train_df, val_df = train_test_split(train_df, test_size=0.2)
		#
		# 	train_val_test_df = cleaned_df
		# 	print(cleaned_df.columns)
		# 	# sys.exit(1)
		#
		# 	# Form np arrays of labels and features.
		# 	train_labels = np.array(train_val_test_df.pop('Class'))
		# 	train_features = np.array(train_val_test_df)
		# 	input_data = np.float32(train_features)
		#
		# 	for ii in range(len(raw_df)):
		# 		print('******', ii, '************')
		# 		infoList = raw_df.loc[ii, ["hour", "nodeID", "has_leak", "leak_group"]].values
		#
		# 		# print(self.groups)
		# 		searchResult = False
		# 		for element in self.selected_groups_list.split(","):
		# 			# print(element)
		# 			if infoList[1] in self.groups[int(element)]:
		# 				searchResult = True
		# 				break
		#
		# 		if searchResult:
		# 			print("send data for : ", infoList[1])
		# 			# print(input_data[ii:ii + 1])
		# 			now = datetime.datetime.now()
		# 			print(now.strftime("%d/%m/%Y %H:%M:%S"))
		# 			print(datetime.datetime.timestamp(now))
		#
		# 			json_str = json.dumps({
		# 				"client": self.name,
		# 				"nodeID": infoList[1],
		# 				"hour": infoList[0],
		# 				"has_leak": str(infoList[2]),
		# 				"leak_group": str(infoList[3]),
		# 				"date": now.strftime("%d/%m/%Y %H:%M:%S"),
		# 				"timestamp": datetime.datetime.timestamp(now),
		# 				"data": input_data[ii:ii + 1].tolist()
		# 			})
		#
		# 			# SEND
		# 			# socket.send(input_data[ii:ii + 1])
		# 			socket.send_json(json_str)
		#
		# 			time.sleep(0.1)

		client.loop_forever()


interrupted = False
def signal_handler(signum, frame):
	global interrupted
	interrupted = True

class SERVER_Process:
# class SERVER_Process(Process):

	def __init__(self, name, wn, smart_sensor_junctions, verbose, logFileName, server_notify_ip):
		Process.__init__(self)
		self.name = name
		self.wn = wn
		self.verbose = verbose
		self.logFileName = logFileName
		self.server_notify_ip = server_notify_ip
		self.smart_sensor_junctions = smart_sensor_junctions

	def run(self):
		print("######### START SERVER ##########")
		print ("SERVER NAME " + self.name)
		print("######### ############ ##########")
		# # Lock on
		# lock.acquire()
		# # Free lock
		# lock.release()

		# Create CSV log
		now = datetime.datetime.now()
		print(now.strftime("%d/%m/%Y %H:%M:%S"))
		log_filename_complete = self.logFileName + "_" + now.strftime("%d%m%Y_%H%M%S") + ".csv"
		out_log = open(log_filename_complete, "w", newline='', encoding='utf-8')
		writer_log = csv.writer(out_log, delimiter=';')
		header = ["startTime", "edgeTime", "cloudTime", "notificationTime", "clientName", "hasLeak", "leakGroup", "prediction"]
		writer_log.writerow(header)


		dataset_tflite_path = "tensorflow_group_datasets/model/tflite/"
		model_tflite_filename = "model_leak_group" + str(leak_group) + "_train_node_" + str(leak_group_model) + ".tflite"

		# verify Tensorflow lite
		interpreter = tf.lite.Interpreter(model_path=dataset_tflite_path + model_tflite_filename)
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()

		print("tf lite float")

		print(input_details)
		print(output_details)

		try:
			#notify
			context_notify = zmq.Context()
			context_notify = context_notify.socket(zmq.REQ)
			context_notify.connect('tcp://' + self.server_notify_ip + ':5558')



			# server
			context = zmq.Context()
			# socket = context.socket(zmq.REP)
			socket = context.socket(zmq.PULL)
			socket.bind('tcp://0.0.0.0:5557')
			print("wait msg")

			# or you can use a custom handler
			# counter = 0
			# signal.signal(signal.SIGINT, signal_handler)
			while True:
				try:
					# msg = socket.recv(zmq.NOBLOCK)
					# print("Ready to receive .... ")
					msg_json_str = socket.recv_json(zmq.NOBLOCK)
					if msg_json_str:
						# print(msg_json_str)
						# socket.send('ah ha!'.encode("utf8"))
						msg_dict = json.loads(msg_json_str)
						# print(msg_dict)
						# print(msg_dict["data"])
						input_data = np.array(msg_dict["data"])
						# print(input_data)

						# %%
						# scaler = StandardScaler()
						# train_features = scaler.fit_transform(train_features)

						# test_predictions_baseline = loaded_model.predict(test_features, batch_size=BATCH_SIZE)
						# cm = confusion_matrix(test_labels, test_predictions_baseline > p)
						# print(cm)

						# print("input data")
						input_data = np.float32(input_data)
						# print(input_data)

						interpreter.set_tensor(input_details[0]['index'], [input_data])
						interpreter.invoke()
						preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])

						if self.verbose:
							print('Predicted:', preds_tf_lite)


						if preds_tf_lite > p_threshold:
							leakage_prediction = 1
							if self.verbose:
								print("leakage detected!!! on : ", msg_dict["leak_group"])
						else:
							leakage_prediction = 0

						startTime = float(msg_dict["tmst"])/1000000
						edgeTime = msg_dict["timestamp"]
						now = datetime.datetime.now()
						cloudTime = datetime.datetime.timestamp(now)
						# print(cloudTime)

						context_notify.send_json(json.dumps({"ACK": True}))
						msg_json_str = context_notify.recv_json()
						if msg_json_str:
							if self.verbose:
								print(msg_json_str)

						now = datetime.datetime.now()
						notificationTime = datetime.datetime.timestamp(now)
						# print(notificationTime)

						out_row = [startTime, edgeTime, cloudTime, notificationTime, msg_dict["client"], int(msg_dict["has_leak"]=="True"), int(msg_dict["leak_group"]=="True"), leakage_prediction]
						writer_log.writerow(out_row)

				except zmq.ZMQError:
					pass
				# counter += 1
				# if interrupted:
				# 	print("W: interrupt received, killing server…")
				# 	break


		# writing the different exception class to catch/ handle the exception
		except EOFError:
			print('Hello user it is EOF exception, please enter something and run me again')
		except KeyboardInterrupt:
			print('trigger ctrl-c button in servere')
		# If both the above exception class does not match, else part will get executed
		else:
			print('Hello user there is some format error')

		writer_log.close()


class SERVER_Notify:

	def __init__(self, name, verbose):
		Process.__init__(self)
		self.name = name
		self.verbose = verbose

	def run(self):
		print("######### START NOTIFY SERVER ##########")
		print("NOTIFY SERVER NAME " + self.name)
		print("######### ############ ##########")

		try:
			# server
			context = zmq.Context()
			socket = context.socket(zmq.REP)
			socket.bind('tcp://0.0.0.0:5558')
			print("wait msg")

			while True:
				try:
					msg_json_str = socket.recv_json(zmq.NOBLOCK)
					if msg_json_str:
						if self.verbose:
							print(msg_json_str)
							#msg_dict = json.loads(msg_json_str)
							#input_data = np.array(msg_dict["data"])
							# print(input_data)
						socket.send_json(json.dumps({"ACK": True}))
				except zmq.ZMQError:
					pass

		# writing the different exception class to catch/ handle the exception
		except EOFError:
			print('Hello user it is EOF exception, please enter something and run me again')
		except KeyboardInterrupt:
			print('trigger ctrl-c button in servere')
		# If both the above exception class does not match, else part will get executed
		else:
			print('Hello user there is some format error')



if __name__ == "__main__":
	verbose = False
	if args.verbose:
		verbose = True

	try:

		if args.notify:
			proc3 = SERVER_Notify(args.notify, verbose)
			proc3.run()

		if args.server:
			proc1 = SERVER_Process(args.server, "1", "2", verbose, args.logFileName, args.serverNotifyIp)
			# proc1.start()
			# proc1.join()
			proc1.run()

		if args.client:
			proc2 = CLIENT_Process(args.client, "3", args.gateway, args.mqttBroker, args.serverIp, verbose)
			# proc2.start()
			# proc2.join()
			proc2.run()





	# writing the different exception class to catch/ handle the exception
	except EOFError:
		print('Hello user it is EOF exception, please enter something and run me again')
	except KeyboardInterrupt:
		# if args.server:
		# 	proc1.kill()
		#
		# if args.client:
		# 	proc2.kill()

		print('trigger ctrl-c button in main')
		# If both the above exception class does not match, else part will get executed
	else:
		print('Hello user there is some format error')






if False: #def client():

	#%%
	# scaler = StandardScaler()
	# train_features = scaler.fit_transform(train_features)

	# test_predictions_baseline = loaded_model.predict(test_features, batch_size=BATCH_SIZE)
	p=0.3
	# cm = confusion_matrix(test_labels, test_predictions_baseline > p)
	# print(cm)

	dataset_tflite_path = "tensorflow_group_datasets/model/tflite/"
	model_tflite_filename = "model_leak_group" + str(leak_group) + "_train_node_" + str(leak_group_model) + ".tflite"

	# verify Tensorflow lite
	interpreter = tf.lite.Interpreter(model_path=dataset_tflite_path + model_tflite_filename)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	print("tf lite float")

	print(input_details)
	print(output_details)

	# for i in range(len(validation_data_set)):
	#     interpreter.set_tensor(input_details[0]['index'], validation_data_set[i])
	#     interpreter.invoke()
	#     preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
	#     print('Predicted:', decode_predictions(preds_tf_lite, top=3)[0])
	#
	#
	input_data = np.float32(train_features)

	for ii in range(180):

		print('******', ii, '************')
		print(raw_df.loc[ii, ["hour", "nodeID", "has_leak"]].values)

		interpreter.set_tensor(input_details[0]['index'], input_data[ii:ii + 1])
		interpreter.invoke()

		preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
		print('Predicted:', preds_tf_lite)

		# print(current_prediction)
		# if not current_prediction == current_label:
		# 	sys.exit(1)

		time.sleep(3)





# dataset_path = 'tensorflow_group_datasets/one_res_small/0_no_leaks_rand_base_demand/'
#
# out_filename = '1M_one_res_small_leaks_ordered_group_0_node_0_0164_merged.csv'
# raw_df1 = pd.read_csv(dataset_path+out_filename, delimiter=";")
#
# EPOCHS = 100
# BATCH_SIZE = 2048
#
#
# # tf.keras.utils.plot_model(loaded_model, to_file='model_plot.png', show_shapes=True)
#
# # tf.keras.utils.plot_model(
# #     loaded_model,
# #     to_file='tensorflow_group_datasets/model_plot.png',
# #     show_shapes=True,
# #     show_dtype=False,
# #     show_layer_names=True,
# #     rankdir='TB',
# #     expand_nested=False,
# #     dpi=96,
# #     layer_range=None,
# #     show_layer_activations=False,
# # )
#
# leak_area = "0164" # "0246" #"0164"
# # 1-->7 2-->7 3-->6 4-->2 5-->1 6-->7 7-->7
# leak_group = 5
# leak_group_model = 1
#
# dataset_path = "tensorflow_group_datasets/model/h5/"
# model_filename = "model_leak_group"+str(leak_group)+"_train_node_"+str(leak_group_model)+".h5"
# loaded_model = tf.keras.models.load_model(dataset_path+model_filename)
#
# dataset_path = 'tensorflow_group_datasets/one_res_small/1_at_82_leaks_rand_base_demand/'
#
# # plt.figure(figsize=(5, 5))
# fig = plt.figure()
#
#
# for leak_node in range(1,10,1):
# 	print("leak node : ", leak_node)
# 	out_filename = "1M_one_res_small_leaks_ordered_group_"+str(leak_group)+"_node_"+str(leak_node)+"_"+leak_area+"_merged.csv"
# 	raw_df2 = pd.read_csv(dataset_path+out_filename, delimiter=";")
#
# 	# Appending multiple DataFrame
# 	raw_df = pd.concat([raw_df1, raw_df2])
# 	raw_df.reset_index(drop=True, inplace=True)
#
# 	cleaned_df = raw_df.copy()
# 	pop_col = ['hour', 'nodeID', 'node_type', 'leak_area_value', 'leak_discharge_value', 'leak_demand_value', 'has_leak' ]
#
# 	cleaned_df = cleaned_df.drop(pop_col, axis=1)
# 	cleaned_df.rename(columns = {'leak_group':'Class'}, inplace = True)
#
# 	#%%
# 	neg, pos = np.bincount(cleaned_df['Class'])
# 	total = neg + pos
# 	print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
# 		total, pos, 100 * pos / total))
#
# 	#%%
# 	# Use a utility from sklearn to split and shuffle your dataset.
# 	train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
# 	train_df, val_df = train_test_split(train_df, test_size=0.2)
#
# 	# Form np arrays of labels and features.
# 	train_labels = np.array(train_df.pop('Class'))
# 	bool_train_labels = train_labels != 0
# 	val_labels = np.array(val_df.pop('Class'))
# 	test_labels = np.array(test_df.pop('Class'))
#
# 	train_features = np.array(train_df)
# 	val_features = np.array(val_df)
# 	test_features = np.array(test_df)
#
# 	#%%
# 	scaler = StandardScaler()
# 	train_features = scaler.fit_transform(train_features)
# 	val_features = scaler.transform(val_features)
# 	test_features = scaler.transform(test_features)
#
# 	#%%
# 	# pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
# 	# neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)
#
#
# 	# baseline_results =loaded_model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
# 	# for name, value in zip(loaded_model.metrics_names, baseline_results):
# 	# 	if name == "loss":
# 	# 		loss = value
# 	# 		print(name, ': ', value)
# 	# 	if name == "accuracy":
# 	# 		acc = value
# 	# 	print(name, ': ', value)
# 	# print()
# 	#
# 	#
# 	# baseline_results =loaded_model.evaluate(val_features, val_labels, batch_size=BATCH_SIZE, verbose=0)
# 	# for name, value in zip(loaded_model.metrics_names, baseline_results):
# 	# 	if name == "loss":
# 	# 		loss = value
# 	# 		print(name, ': ', value)
# 	# 	if name == "accuracy":
# 	# 		acc = value
# 	# 	print(name, ': ', value)
# 	# print()
#
# 	baseline_results =loaded_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
# 	for name, value in zip(loaded_model.metrics_names, baseline_results):
# 		# print(name, ': ', value)
# 		if name == "loss":
# 			loss = value
# 			print(name, ': ', value)
# 		if name == "accuracy":
# 			acc = value
# 			print(name, ': ', value)
#
# 	test_predictions_baseline = loaded_model.predict(test_features, batch_size=BATCH_SIZE)
#
# 	# plot_cm(test_labels, test_predictions_baseline, "leak node : "+str(leak_node))
# 	p=0.3 #0.5
#
# 	cm = confusion_matrix(test_labels, test_predictions_baseline > p)
# 	print("******************************")
# 	print(leak_node, ",", loss, ",", acc, ",", cm[0][0], ",", cm[0][1], ",", cm[1][0], ",", cm[1][1])
# 	# plt.figure(figsize=(5, 5))
# 	ax1 = fig.add_subplot(3, 3, leak_node)
# 	sns.heatmap(cm, annot=True, fmt="d")
#
# 	plt.title("leak node : "+str(leak_node))
#
# 	if leak_node in [1, 4, 7]:
# 		plt.ylabel('Actual label')
# 	if leak_node in [7, 8, 9]:
# 		plt.xlabel('Predicted label')
#
# 	# plt.show()
# 	#create scalar file
# 	#
# 	# not_scaled_df = raw_df2.copy()
# 	# pop_col_not_scaled_df = ['base_demand', 'demand_value', 'head_value', 'pressure_value', 'x_pos', 'y_pos',
# 	# 						 'flow_demand_in', 'demand_0', 'head_0', 'pressure_0',
# 	# 						 'demand_1', 'head_1', 'pressure_1', 'demand_2', 'head_2', 'pressure_2',
# 	# 						 'demand_3', 'head_3', 'pressure_3', 'demand_4', 'head_4', 'pressure_4',
# 	# 						 'demand_5', 'head_5', 'pressure_5', 'demand_6', 'head_6', 'pressure_6',
# 	# 						 'demand_7', 'head_7', 'pressure_7', 'demand_8', 'head_8', 'pressure_8',
# 	# 						 'demand_9', 'head_9', 'pressure_9', 'flow_demand_out']
# 	# not_scaled_df = not_scaled_df.drop(pop_col_not_scaled_df, axis=1)
# 	#
# 	# scaled_df = raw_df2.copy()
# 	# pop_col_scaled_df = ['hour', 'nodeID', 'node_type', 'leak_area_value', 'leak_discharge_value', 'leak_demand_value', 'has_leak', 'leak_group']
# 	# scaled_df = scaled_df.drop(pop_col_scaled_df, axis=1)
# 	# scaled_columns_names = scaled_df.columns
# 	# columns_names_scalar = []
# 	# for kk in range(len(scaled_columns_names)):
# 	# 	columns_names_scalar.append(scaled_columns_names[kk]+"_scalar")
# 	# scaled_array = np.array(scaled_df)
# 	# cleaned_df_for_scalar_new_columns = scaler.transform(scaled_array)
# 	# cleaned_df_for_scalar_new_columns = pd.DataFrame(data=cleaned_df_for_scalar_new_columns, columns=columns_names_scalar)
# 	#
# 	# raw_df_for_scalar_final = pd.concat([not_scaled_df, cleaned_df_for_scalar_new_columns], axis=1)
# 	#
# 	# out_filename = "1M_one_res_small_leaks_ordered_group_"+str(leak_group)+"_node_"+str(leak_node)+"_"+leak_area+"_merged_scalar.csv"
# 	# raw_df_for_scalar_final.to_csv(dataset_path+out_filename, float_format='%.8f', index=False, sep=';')
#
# fig_output_filename = "tensorflow_group_datasets/fig/" + "model_leak_group" + str(leak_group) + "_train_node_" + str(leak_group_model) + "_from_python_2.png"
# plt.savefig(fig_output_filename, dpi=300, bbox_inches="tight")
#
