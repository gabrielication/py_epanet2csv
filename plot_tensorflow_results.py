import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
# from itertools import combinations
# from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
# import math


# https://stackabuse.com/change-tick-frequency-in-matplotlib/
# https://sparkbyexamples.com/pandas/pandas-select-dataframe-rows-based-on-column-values/

def delta_plot_after_tensorflow_analysis(path, bool_base_demand, bool_pressure_value, y_data, save=True, show=False):
	data = pd.read_csv(path)

	data = data.loc[(data['base_demand'] == bool_base_demand) & (data['pressure_value'] == bool_pressure_value)]

	x = ["no leak", "1 leak", "1/8 leaks", "1/4 leaks", "1/2 leaks"]
	y = data[y_data]

	time_period = path[0:2]

	plot_title = time_period+" with "
	output_filename = time_period+"_"+y_data+"_with_"
	# output_filename = "1W_delta_loss_with_bdem_press.png"

	if(bool_base_demand == True and bool_pressure_value == False):
		plot_title = plot_title+"base_demand"
		output_filename = output_filename+"base_demand"
	elif(bool_base_demand == False and bool_pressure_value == True):
		plot_title = plot_title + "pressure"
		output_filename = output_filename + "pressure"
	elif (bool_base_demand == True and bool_pressure_value == True):
		plot_title = plot_title + "base_demand and pressure"
		output_filename = output_filename + "base_demand_pressure"

	output_filename = output_filename+".png"

	# plt.yticks(np.arange(0, max(y), 0.00001))

	# fig = plt.figure()
	# ax = fig.add_subplot()

	plt.plot(x, y)
	plt.title(plot_title)

	# plt.plot(x,y,'ro')

	# for i,j in zip(x,y):
	#     ax.annotate(str(j), xy=(i,j), xytext=(10,20), textcoords='offset points')

	plt.xlabel('Dataset')
	# plt.ylabel('Delta Loss')

	plt.ylabel(y_data)

	# plt.legend()
	plt.grid(True)

	if(save):
		plt.savefig(output_filename, dpi=300, bbox_inches = "tight")
		print("Saved to: " + output_filename)

	if(show):
		plt.show()

	plt.clf()


def simple_features_combination_plot_after_tensorflow_analysis(path, dataset, y_data, save=True, show=False):
	data = pd.read_csv(path)

	data = data.loc[(data['dataset'] == dataset)]

	x = ["base_demand","pressure","bd+press"]
	y = data[y_data]

	time_period = path[0:2]

	plot_title = time_period+" "+y_data+" with feat combinations"

	output_filename = time_period+"_"+y_data+"_with_feat_combinations"
	output_filename = output_filename+".png"

	plt.plot(x, y)
	plt.title(plot_title)

	plt.xlabel("features")
	plt.ylabel(y_data)

	plt.grid(True)

	if(save):
		plt.savefig(output_filename, dpi=300, bbox_inches = "tight")
		print("Saved to: "+output_filename)

	if(show):
		plt.show()

	plt.clf()

def polygon_under_graph(x, y):
	"""
	Construct the vertex list which defines the polygon filling the space under
	the (x, y) line graph. This assumes x is in ascending order.
	"""
	return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

def plot_user_demand(path, save=False, show=True):

	# Fixing random state for reproducibility
	# np.random.seed(19680801)

	fig = plt.figure()

	for ii in range(1,3):

		ax1 = fig.add_subplot(1, 2, ii, projection='3d')
		# create two subplots with the shared x and y axes
		# fig, (ax1, ax2) = plt.subplots(1, 2, projection='3d', sharex=False, sharey=False)

		data = pd.read_csv(path)

		colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
		# yticks = [7, 6, 5, 4, 3, 2, 1, 0]
		yticks = [0,1,2,3,4,5,6,7]

		nodeList = data.nodeID.unique()
		print(nodeList)

		nodeIndex = 0
		nodeSelected = []
		for iterationIndex in range(4):
			dataToPlot = data.loc[(data['nodeID'] == nodeList[nodeIndex])]
			nodeSelected.append(nodeList[nodeIndex])

			#hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand
			baseDemandToPlot = dataToPlot['base_demand'].values #*  3.785412 * 60
			demandToPlot = dataToPlot['demand_value'].values #*  3.785412 * 60

			# for c, k in zip(colors, yticks):
			c = colors[iterationIndex]; k = yticks[iterationIndex];

			# create data for the y=k 'layer'.
			xs = np.arange(len(demandToPlot))
			# ys = demandToPlot

			if ii == 1:
				ys = baseDemandToPlot
			else:
				ys = demandToPlot

			# You can provide either a single color or an array with the same length as
			# xs and ys. To demonstrate this, we color the first bar of each set cyan.
			cs = [c] * len(xs)
			# cs[0] = 'c'

			# Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
			ax1.bar(xs, ys, k, zdir='y', color=cs, alpha=0.8)

			# ls = LightSource(270, 45)
			# # To use a custom hillshading mode, override the built-in shading and pass
			# # in the rgb colors of the shaded surface calculated from "shade".
			# rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
			# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
			#                        linewidth=0, antialiased=False, shade=False)

			nodeIndex += 15
			# if nodeIndex>len(colors)-1:
			# if nodeIndex > 3:
			#     break

		ax1.set_xlabel('Hours')
		ax1.set_ylabel('Node')
		if ii == 1:
			ax1.set_zlabel('Base demand [G/M]')
			# ax1.set_zlabel('Base demand [l/H]')
		else:
			ax1.set_zlabel('Demand value [G/M]')
			# ax1.set_zlabel('Demand value [l/H]')

		# On the y axis let's only label the discrete values that we have data for.
		# ax.set_yticks(yticks)
		ax1.set_yticks([1,2,3,4])
		# ax1.set_ylabel(nodeSelected)
		# ax1.set_yticklabels(ax1.get_yticks(), rotation=30)
		ax1.set_yticklabels(nodeSelected, rotation=90)

	# bool_base_demand = True
	# bool_pressure_value = False
	# data = data.loc[(data['base_demand'] == bool_base_demand) & (data['pressure_value'] == bool_pressure_value)]
	#
	# # x = ["no leak", "1 leak", "1/8 leaks", "1/4 leaks", "1/2 leaks"]
	# # y = data[y_data]
	#
	# time_period = path[0:2]
	#
	# plot_title = time_period + " with "
	output_filename = "one_res_small_no_leaks_node_demand" + "_" #+ y_data + "_with_"
	# # output_filename = "1W_delta_loss_with_bdem_press.png"
	#
	# if (bool_base_demand == True and bool_pressure_value == False):
	#     plot_title = plot_title + "base_demand"
	#     output_filename = output_filename + "base_demand"
	# elif (bool_base_demand == False and bool_pressure_value == True):
	#     plot_title = plot_title + "pressure"
	#     output_filename = output_filename + "pressure"
	# elif (bool_base_demand == True and bool_pressure_value == True):
	#     plot_title = plot_title + "base_demand and pressure"
	#     output_filename = output_filename + "base_demand_pressure"
	#
	output_filename = "fig/" + output_filename + ".png"
	#
	# # plt.yticks(np.arange(0, max(y), 0.00001))
	#
	# # fig = plt.figure()
	# # ax = fig.add_subplot()
	#
	# # plt.plot(x, y)
	# plt.title(plot_title)
	#
	# # plt.plot(x,y,'ro')
	#
	# # for i,j in zip(x,y):
	# #     ax.annotate(str(j), xy=(i,j), xytext=(10,20), textcoords='offset points')
	#
	# plt.xlabel('Dataset')
	# # plt.ylabel('Delta Loss')
	#
	# # plt.ylabel(y_data)
	# plt.ylabel("user demand")
	#
	# # plt.legend()
	# plt.grid(True)
	#
	if (save):
		plt.savefig(output_filename, dpi=300, bbox_inches="tight")
		print("Saved to: " + output_filename)

	if (show):
		plt.show()

	plt.clf()



def plot_user_demand_v2(path, save=False, show=True):

	# Fixing random state for reproducibility
	# np.random.seed(19680801)



	# create two subplots with the shared x and y axes
	# fig, (ax1, ax2) = plt.subplots(1, 2, projection='3d', sharex=False, sharey=False)

	data = pd.read_csv(path)

	colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
	# yticks = [7, 6, 5, 4, 3, 2, 1, 0]
	yticks = [0,1,2,3,4,5,6,7]

	nodeList = data.nodeID.unique()
	print(nodeList)

	for ii in range(0,4):
		print("FIGURE : ", ii)
		# fig = plt.figure()

		if ii<3:
			fig, axs = plt.subplots(4, 6, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
		else:
			fig, axs = plt.subplots(2, 6, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

		# fig.suptitle('Sharing both axes')

		nodeIndex = 1
		plotIndexRow = 0
		plotIndexColumn = 0
		for node in nodeList[ii*24:(ii+1)*24]:
			print("SUBPLOT node : ", node)
			# ax1 = fig.add_subplot(4, 6, nodeIndex)

			dataToPlot = data.loc[(data['nodeID'] == node)]

			#hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand
			baseDemandToPlot = dataToPlot['base_demand'].values #*  3.785412 * 60
			leakDemandToPlot = dataToPlot['leak_demand_value'].values #*  3.785412 * 60
			demandToPlot = dataToPlot['demand_value'].values #*  3.785412 * 60

			# for c, k in zip(colors, yticks):
			# c = colors[nodeIndex]; k = yticks[nodeIndex];

			# create data for the y=k 'layer'.
			xs = np.arange(len(demandToPlot))

			# You can provide either a single color or an array with the same length as
			# xs and ys. To demonstrate this, we color the first bar of each set cyan.
			# cs = [c] * len(xs)
			# cs[0] = 'c'

			# Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
			# ax1.bar(xs, ys, k, zdir='y', color=cs, alpha=0.8)
			# ax1.plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
			# ax1.plot(xs, demandToPlot, label='Demand value node ' + str(node))

			print(plotIndexRow, ' : ', plotIndexColumn)

			# axs[plotIndexRow, plotIndexColumn].plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
			# axs[plotIndexRow, plotIndexColumn].plot(xs, demandToPlot, label='Demand value node ' + str(node))
			axs[plotIndexRow, plotIndexColumn].plot(xs, baseDemandToPlot, label='Base Demand node')
			axs[plotIndexRow, plotIndexColumn].plot(xs, demandToPlot, label='Demand value node ')
			axs[plotIndexRow, plotIndexColumn].plot(xs, leakDemandToPlot, label='Leak Demand value node ')

			# ls = LightSource(270, 45)
			# # To use a custom hillshading mode, override the built-in shading and pass
			# # in the rgb colors of the shaded surface calculated from "shade".
			# rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
			# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
			#                        linewidth=0, antialiased=False, shade=False)

			axs[plotIndexRow, plotIndexColumn].set_xlabel('Hours')
			if plotIndexColumn == 0:
				axs[plotIndexRow, plotIndexColumn].set_ylabel('[G/M]')

			plotIndexColumn += 1
			if plotIndexColumn > 5:
				plotIndexRow += 1
				plotIndexColumn = 0


			nodeIndex += 1
			# if nodeIndex>len(colors)-1:
			if nodeIndex > 24:
				break

		axs[0, 5].legend()

		# if ii == 1:
		#     ax1.set_zlabel('Base demand [G/M]')
		# else:
		#     ax1.set_zlabel('Demand value [G/M]')

		# On the y axis let's only label the discrete values that we have data for.
		# ax.set_yticks(yticks)
		# ax1.set_yticks([1,2,3,4])

		# bool_base_demand = True
		# bool_pressure_value = False
		# data = data.loc[(data['base_demand'] == bool_base_demand) & (data['pressure_value'] == bool_pressure_value)]
		#
		# # x = ["no leak", "1 leak", "1/8 leaks", "1/4 leaks", "1/2 leaks"]
		# # y = data[y_data]
		#
		# time_period = path[0:2]
		#
		# plot_title = time_period + " with "

		# output_filename = "one_res_small_no_leaks_node_demand_base_value_" + str(ii) + "_" #+ y_data + "_with_"
		output_filename = path.replace("/", "-") + str(ii)

		# # output_filename = "1W_delta_loss_with_bdem_press.png"
		#
		# if (bool_base_demand == True and bool_pressure_value == False):
		#     plot_title = plot_title + "base_demand"
		#     output_filename = output_filename + "base_demand"
		# elif (bool_base_demand == False and bool_pressure_value == True):
		#     plot_title = plot_title + "pressure"
		#     output_filename = output_filename + "pressure"
		# elif (bool_base_demand == True and bool_pressure_value == True):
		#     plot_title = plot_title + "base_demand and pressure"
		#     output_filename = output_filename + "base_demand_pressure"
		#
		output_filename = "fig2/" + output_filename + ".png"
		#
		# # plt.yticks(np.arange(0, max(y), 0.00001))
		#
		# # fig = plt.figure()
		# # ax = fig.add_subplot()
		#
		# # plt.plot(x, y)
		# plt.title(plot_title)
		#
		# # plt.plot(x,y,'ro')
		#
		# # for i,j in zip(x,y):
		# #     ax.annotate(str(j), xy=(i,j), xytext=(10,20), textcoords='offset points')
		#
		# plt.xlabel('Dataset')
		# # plt.ylabel('Delta Loss')
		#
		# # plt.ylabel(y_data)
		# plt.ylabel("user demand")
		#
		# # plt.legend()
		# plt.grid(True)
		#
		if (save):
			plt.savefig(output_filename, dpi=300, bbox_inches="tight")
			print("Saved to: " + output_filename)

		if (show):
			plt.show()

		plt.clf()


def plot_hist_user_demand(path, save=False, show=True):

	# Fixing random state for reproducibility
	# np.random.seed(19680801)

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	data = pd.read_csv(path)

	colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
	# yticks = [7, 6, 5, 4, 3, 2, 1, 0]
	yticks = [0,1,2,3,4,5,6,7]

	nodeList = data.nodeID.unique()
	nodeIndex = 0
	for node in nodeList:
		dataToPlot = data.loc[(data['nodeID'] == node)]
		#hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand
		demandToPlot = dataToPlot['demand_value'].values
		baseDemandToPlot = dataToPlot['base_demand'].values

		# count, division = np.histogram(baseDemandToPlot, bins=40, range=[0, 0.01], density=False)
		# count, division = np.histogram(baseDemandToPlot, bins=20, density=False)
		count, division = np.histogram(baseDemandToPlot, bins=20)
		print(count)
		print(division)

		width = 0.7 * (division[1] - division[0])
		center = (division[:-1] + division[1:]) / 2
		# plt.bar(center, count, align='center', width=width)
		# plt.show()
		#
		# sys.exit(1)

		# for c, k in zip(colors, yticks):
		c = colors[nodeIndex]; k = yticks[nodeIndex];

		# create data for the y=k 'layer'.
		# xs = np.arange(len(demandToPlot))
		xs = np.arange(20)
		# xs = division[:-1]


		# ys = demandToPlot
		# ys = baseDemandToPlot
		ys = count

		# You can provide either a single color or an array with the same length as
		# xs and ys. To demonstrate this, we color the first bar of each set cyan.
		cs = [c] * len(xs)
		# cs[0] = 'c'

		# Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
		ax.bar(xs, ys, k, zdir='y', color=cs, alpha=0.8)
		# ls = LightSource(270, 45)
		# # To use a custom hillshading mode, override the built-in shading and pass
		# # in the rgb colors of the shaded surface calculated from "shade".
		# rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
		# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
		#                        linewidth=0, antialiased=False, shade=False)

		nodeIndex += 1
		# if nodeIndex>len(colors)-1:
		if nodeIndex > 3:
			break

	ax.set_xticklabels(center.round(6), rotation=65)

	ax.set_xlabel('Demand center bin')
	ax.set_ylabel('Node')
	ax.set_zlabel('Frequency')

	# On the y axis let's only label the discrete values that we have data for.
	# ax.set_yticks(yticks)
	ax.set_yticks([1,2,3,4])


	# bool_base_demand = True
	# bool_pressure_value = False
	# data = data.loc[(data['base_demand'] == bool_base_demand) & (data['pressure_value'] == bool_pressure_value)]
	#
	# # x = ["no leak", "1 leak", "1/8 leaks", "1/4 leaks", "1/2 leaks"]
	# # y = data[y_data]
	#
	# time_period = path[0:2]
	#
	# plot_title = time_period + " with "
	output_filename = "one_res_small_no_leaks_hist_node_demand" + "_" #+ y_data + "_with_"

	# # output_filename = "1W_delta_loss_with_bdem_press.png"
	#
	# if (bool_base_demand == True and bool_pressure_value == False):
	#     plot_title = plot_title + "base_demand"
	#     output_filename = output_filename + "base_demand"
	# elif (bool_base_demand == False and bool_pressure_value == True):
	#     plot_title = plot_title + "pressure"
	#     output_filename = output_filename + "pressure"
	# elif (bool_base_demand == True and bool_pressure_value == True):
	#     plot_title = plot_title + "base_demand and pressure"
	#     output_filename = output_filename + "base_demand_pressure"
	#
	output_filename = "fig/" + output_filename + ".png"
	#
	# # plt.yticks(np.arange(0, max(y), 0.00001))
	#
	# # fig = plt.figure()
	# # ax = fig.add_subplot()
	#
	# # plt.plot(x, y)
	# plt.title(plot_title)
	#
	# # plt.plot(x,y,'ro')
	#
	# # for i,j in zip(x,y):
	# #     ax.annotate(str(j), xy=(i,j), xytext=(10,20), textcoords='offset points')
	#
	# plt.xlabel('Dataset')
	# # plt.ylabel('Delta Loss')
	#
	# # plt.ylabel(y_data)
	# plt.ylabel("user demand")
	#
	# # plt.legend()
	# plt.grid(True)
	#
	if (save):
		plt.savefig(output_filename, dpi=300, bbox_inches="tight")
		print("Saved to: " + output_filename)

	if (show):
		plt.show()
	#
	# plt.clf()


def plot_cumulative_hist_user_demand(path, save=False, show=True):

	# Fixing random state for reproducibility
	# np.random.seed(19680801)

	fig = plt.figure()
	ax = fig.add_subplot()

	data = pd.read_csv(path)

	colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
	# yticks = [7, 6, 5, 4, 3, 2, 1, 0]
	yticks = [0,1,2,3,4,5,6,7]

	nodeList = data.nodeID.unique()
	nodeIndex = 0
	for node in nodeList:
		print(node)
		dataToPlot = data.loc[(data['nodeID'] == node)]
		#hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand

		demandToPlot = dataToPlot['demand_value'].values
		# baseDemandToPlot = dataToPlot['base_demand'].values

		selectedBins = 40 #20
		count, division = np.histogram(demandToPlot, bins=selectedBins, range=[0, 0.00009], density=False)
		# count, division = np.histogram(baseDemandToPlot, bins=selectedBins, density=False)


		print(count)
		# print(division)

		width = 0.7 * (division[1] - division[0])
		center = (division[:-1] + division[1:]) / 2
		# plt.bar(center, count, align='center', width=width)
		# plt.show()
		#

		dx = division[1] - division[0]
		F1 = np.cumsum(count) / np.sum(count)
		# print(F1)
		# method 2
		# X2 = np.sort(baseDemandToPlot)
		# F2 = np.array( range(len(demandToPlot))) / float(len(demandToPlot))

		plt.plot(division[1:], F1)
		# plt.plot(X2, F2)
		# sys.exit(1)


		nodeIndex += 1
		# if nodeIndex>len(colors)-1:
		# if nodeIndex > 10: #81:
		#     break

	# ax.set_xticklabels(center.round(6), rotation=65)

	ax.set_xlabel('Demand [bin ('+str(selectedBins)+')] [G/M]')
	ax.set_ylabel('CDF')


	# bool_base_demand = True
	# bool_pressure_value = False
	# data = data.loc[(data['base_demand'] == bool_base_demand) & (data['pressure_value'] == bool_pressure_value)]
	#
	# # x = ["no leak", "1 leak", "1/8 leaks", "1/4 leaks", "1/2 leaks"]
	# # y = data[y_data]
	#
	# time_period = path[0:2]
	#
	# plot_title = time_period + " with "
	# output_filename = "one_res_small_no_leaks_cdf_node_demand" + "_" #+ y_data + "_with_"
	output_filename = path.replace("/", "-")
	# # output_filename = "1W_delta_loss_with_bdem_press.png"
	#
	# if (bool_base_demand == True and bool_pressure_value == False):
	#     plot_title = plot_title + "base_demand"
	#     output_filename = output_filename + "base_demand"
	# elif (bool_base_demand == False and bool_pressure_value == True):
	#     plot_title = plot_title + "pressure"
	#     output_filename = output_filename + "pressure"
	# elif (bool_base_demand == True and bool_pressure_value == True):
	#     plot_title = plot_title + "base_demand and pressure"
	#     output_filename = output_filename + "base_demand_pressure"
	#
	output_filename = "fig/" + output_filename + ".png"
	#
	# # plt.yticks(np.arange(0, max(y), 0.00001))
	#
	# # fig = plt.figure()
	# # ax = fig.add_subplot()
	#
	# # plt.plot(x, y)
	# plt.title(plot_title)
	#
	# # plt.plot(x,y,'ro')
	#
	# # for i,j in zip(x,y):
	# #     ax.annotate(str(j), xy=(i,j), xytext=(10,20), textcoords='offset points')
	#
	# plt.xlabel('Dataset')
	# # plt.ylabel('Delta Loss')
	#
	# # plt.ylabel(y_data)
	# plt.ylabel("user demand")
	#
	# # plt.legend()
	# plt.grid(True)
	#
	if (save):
		plt.savefig(output_filename, dpi=300, bbox_inches="tight")
		print("Saved to: " + output_filename)

	if (show):
		plt.show()

	plt.clf()




def plot_model_analysis(path, save=False, show=True):

	# Fixing random state for reproducibility
	# np.random.seed(19680801)

	# create two subplots with the shared x and y axes
	# fig, (ax1, ax2) = plt.subplots(1, 2, projection='3d', sharex=False, sharey=False)

	colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']
	# yticks = [7, 6, 5, 4, 3, 2, 1, 0]
	yticks = [0,1,2,3,4,5,6,7]


	for ii in range(1,8):
		print("FIGURE : ", ii)
		# fig = plt.figure()

		complete_path = path + str(ii) + ".csv"
		data = pd.read_csv(complete_path, delimiter=";")
		print("group node model : ", data.loc[ii, 'leak_group_model'])

		# if ii<3:
		#     fig, axs = plt.subplots(4, 6, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
		# else:
		#     fig, axs = plt.subplots(2, 6, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
		fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
		# fig.suptitle('Sharing both axes')

		nodeIndex = 1
		plotIndexRow = 0
		plotIndexColumn = 0
		for jj in range(1,10):
			print("leak node model: ", data.loc[(jj-1)*9, 'leak_node_model'])
			# ax1 = fig.add_subplot(4, 6, nodeIndex)

			dataToPlot = data.loc[(jj-1)*9:((jj-1)*9)+8]

			#leak_group_model;leak_node_model;leak_node_test;loss;accuracy

			accuracyPlot = dataToPlot['accuracy'].values
			print(accuracyPlot)

			# for c, k in zip(colors, yticks):
			# c = colors[nodeIndex]; k = yticks[nodeIndex];

			# create data for the y=k 'layer'.
			# xs = np.arange(len(demandToPlot))
			xs = dataToPlot['leak_node_test'].values
			print(xs)
			# You can provide either a single color or an array with the same length as
			# xs and ys. To demonstrate this, we color the first bar of each set cyan.
			# cs = [c] * len(xs)
			# cs[0] = 'c'

			# Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
			# ax1.bar(xs, ys, k, zdir='y', color=cs, alpha=0.8)
			# ax1.plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
			# ax1.plot(xs, demandToPlot, label='Demand value node ' + str(node))

			print(plotIndexRow, ' : ', plotIndexColumn)

			# axs[plotIndexRow, plotIndexColumn].plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
			# axs[plotIndexRow, plotIndexColumn].plot(xs, demandToPlot, label='Demand value node ' + str(node))
			axs.plot(xs, accuracyPlot, label="learning node : "+str(data.loc[(jj-1)*9, 'leak_node_model']))

			# ls = LightSource(270, 45)
			# # To use a custom hillshading mode, override the built-in shading and pass
			# # in the rgb colors of the shaded surface calculated from "shade".
			# rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
			# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
			#                        linewidth=0, antialiased=False, shade=False)

		axs.set_xlabel('Node for test')
		axs.set_ylabel('Accuracy')
		axs.set_title("group node model : "+str(data.loc[ii, 'leak_group_model']))
		axs.legend(ncol=2)


		plotIndexColumn += 1
		if plotIndexColumn > 5:
			plotIndexRow += 1
			plotIndexColumn = 0

		axs.axis(ymin=0.7, ymax=1.1)
		# nodeIndex += 1
		# # if nodeIndex>len(colors)-1:
		# if nodeIndex > 24:
		#     break
		#
		# axs[0, 5].legend()

		# if ii == 1:
		#     ax1.set_zlabel('Base demand [G/M]')
		# else:
		#     ax1.set_zlabel('Demand value [G/M]')

		# On the y axis let's only label the discrete values that we have data for.
		# ax.set_yticks(yticks)
		# ax1.set_yticks([1,2,3,4])

		# bool_base_demand = True
		# bool_pressure_value = False
		# data = data.loc[(data['base_demand'] == bool_base_demand) & (data['pressure_value'] == bool_pressure_value)]
		#
		# # x = ["no leak", "1 leak", "1/8 leaks", "1/4 leaks", "1/2 leaks"]
		# # y = data[y_data]
		#
		# time_period = path[0:2]
		#
		# plot_title = time_period + " with "

		# output_filename = "one_res_small_no_leaks_node_demand_base_value_" + str(ii) + "_" #+ y_data + "_with_"
		# output_filename = path.replace("/", "-") + str(ii)

		# # output_filename = "1W_delta_loss_with_bdem_press.png"
		#
		# if (bool_base_demand == True and bool_pressure_value == False):
		#     plot_title = plot_title + "base_demand"
		#     output_filename = output_filename + "base_demand"
		# elif (bool_base_demand == False and bool_pressure_value == True):
		#     plot_title = plot_title + "pressure"
		#     output_filename = output_filename + "pressure"
		# elif (bool_base_demand == True and bool_pressure_value == True):
		#     plot_title = plot_title + "base_demand and pressure"
		#     output_filename = output_filename + "base_demand_pressure"
		#
		output_filename = path + str(ii) + ".png"
		#
		# # plt.yticks(np.arange(0, max(y), 0.00001))
		#
		# # fig = plt.figure()
		# # ax = fig.add_subplot()
		#
		# # plt.plot(x, y)
		# plt.title(plot_title)
		#
		# # plt.plot(x,y,'ro')
		#
		# # for i,j in zip(x,y):
		# #     ax.annotate(str(j), xy=(i,j), xytext=(10,20), textcoords='offset points')
		#
		# plt.xlabel('Dataset')
		# # plt.ylabel('Delta Loss')
		#
		# # plt.ylabel(y_data)
		# plt.ylabel("user demand")
		#
		# # plt.legend()
		# plt.grid(True)
		#
		if (save):
			plt.savefig(output_filename, dpi=300, bbox_inches="tight")
			print("Saved to: " + output_filename)

	if (show):
		plt.show()

	plt.clf()




def plot_lost_demand(data, data_leakage, data_leakage_2, save=False, show=True):





	colors = ['black',  'blue', 'red', 'grey', 'brown',  'orange', 'gold', 'yellow', 'cyan', 'pink']
	nodeList = data.nodeID.unique()
	print(nodeList)

	# cols = ["lost_0", "head_0", "pressure_0", "lost_1", "head_1", "pressure_1", "lost_2", "head_2", "pressure_2",
	#           "lost_3", "head_3", "pressure_3", "lost_4", "head_4", "pressure_4", "lost_5", "head_5", "pressure_5",
	#           "lost_6", "head_6", "pressure_6", "lost_7", "head_7", "pressure_7", "lost_8", "head_8", "pressure_8",
	#           "lost_9", "head_9", "pressure_9"]
	# #hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand


	nodeGroup = data.groupby('nodeID').mean()
	nodeLeakageGroup = data_leakage.groupby('nodeID').mean()
	nodeLeakageGroup2 = data_leakage_2.groupby('nodeID').mean()

	#print network topology
	if True:
		min_x_pos = data["x_pos"].min()
		min_y_pos = data["y_pos"].min()
		colsCoordinates = ["x_pos", "y_pos", "has_leak"]

		fig2, axs2 = plt.subplots(1, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
		colorIndex = 0

		for jj in range(1,76,1):
			# print(jj)
			node = nodeList[jj-1]
			cs = colors[colorIndex]
			lostDemandToPlot = nodeGroup.loc[node, colsCoordinates]

			if jj % 10 == 0:
				axs2.scatter((lostDemandToPlot['x_pos']-min_x_pos)/1000, (lostDemandToPlot['y_pos']-min_y_pos)/1000, marker='o', c=cs, s=30, label = 'Group '+str(colorIndex+1))
				# axs2.annotate(str(nodeGroup.index[jj-1]), xy=(lostDemandToPlot['x_pos'], lostDemandToPlot['y_pos']), xytext=(10,20), textcoords='offset points', fontsize=6)
			else:
				axs2.scatter((lostDemandToPlot['x_pos']-min_x_pos)/1000, (lostDemandToPlot['y_pos']-min_y_pos)/1000, marker='o', c=cs, s=30)
				# axs2.annotate(str(nodeGroup.index[jj-1]), xy=(lostDemandToPlot['x_pos'], lostDemandToPlot['y_pos']), xytext=(10,20), textcoords='offset points', fontsize=6)

			if jj % 10==0:
				colorIndex += 1

		axs2.legend(ncol=4, loc='lower right')
		axs2.set_ylabel('Y [km]')
		axs2.set_xlabel('X [km]')
		axs2.set_ylim([-0.2, 3.57])

		output_filename = "tensorflow_group_datasets/fig/node_network_group_position.png"
		plt.savefig(output_filename, dpi=300, bbox_inches="tight")
		plt.show()


	if False:
		#average base_demand and demand_value for each node
		print("FIGURE : average base_demand and demand_value for each node")

		#1
		# fig = plt.figure()
		fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

		BaseDemandToPlotScenario1 = nodeGroup['base_demand'].values[0:75]
		BaseDemandToPlotScenario2 = nodeLeakageGroup['base_demand'].values[0:75]
		BaseDemandToPlotScenario3 = nodeLeakageGroup2['base_demand'].values[0:75]
		# DemandValueToPlot = nodeGroup['demand_value']
		# PressureValueToPlot = nodeGroup['pressure_value']

		xs = np.arange(len(BaseDemandToPlotScenario1))
		cs = colors[0]
		axs.plot(xs, BaseDemandToPlotScenario1, linestyle='-', color=cs, label='Base demand no leak')
		cs = colors[1]
		axs.plot(xs, BaseDemandToPlotScenario2, linestyle='--', color=cs, label='Base demand leak group 3(4)')
		cs = colors[2]
		axs.plot(xs, BaseDemandToPlotScenario3, linestyle='-.', color=cs, label='Base demand leak group 5(4)')
		axs.set_ylim([0, 0.007])

		axs.set_xlabel('Nodes')
		axs.set_ylabel('[GPM]')
		axs.legend(ncol=2, loc='upper right')

		if (save):
			output_filename = "tensorflow_group_datasets/fig/node_network_group_slope_1.png"
			plt.savefig(output_filename, dpi=300, bbox_inches="tight")
			print("Saved to: " + output_filename)

		#2
		# fig = plt.figure()
		fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

		DemandValueToPlotScenario1 = nodeGroup['demand_value'].values[0:75]
		DemandValueToPlotScenario2 = nodeLeakageGroup['demand_value'].values[0:75]
		DemandValueToPlotScenario3 = nodeLeakageGroup2['demand_value'].values[0:75]

		xs = np.arange(len(DemandValueToPlotScenario1))
		cs = colors[0]
		axs.plot(xs, DemandValueToPlotScenario1, linestyle='-', color=cs, label='demand value no leak')
		cs = colors[1]
		axs.plot(xs, DemandValueToPlotScenario2, linestyle='--', color=cs, label='demand value leak group 3(4)')
		cs = colors[2]
		axs.plot(xs, DemandValueToPlotScenario3, linestyle='-.', color=cs, label='demand value leak group 5(4)')
		axs.set_ylim([0, 0.007])


		for kk in range(10, 71, 10):
			xs = np.ones(8)*kk
			ys = np.arange(0,0.008,0.001)
			cs = colors[3]
			axs.plot(xs, ys, linestyle='dotted', color=cs)

		axs.set_xlabel('Nodes')
		axs.set_ylabel('[GPM]')
		axs.legend(ncol=2, loc='upper right')



		if (save):
			output_filename = "tensorflow_group_datasets/fig/node_network_group_slope_2.png"
			plt.savefig(output_filename, dpi=300, bbox_inches="tight")
			print("Saved to: " + output_filename)

		#3
		# fig = plt.figure()
		fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

		PressureValueToPlotScenario1 = nodeGroup['pressure_value'].values[0:75]
		PressureValueToPlotScenario2 = nodeLeakageGroup['pressure_value'].values[0:75]
		PressureValueToPlotScenario3 = nodeLeakageGroup2['pressure_value'].values[0:75]

		xs = np.arange(len(PressureValueToPlotScenario1))
		cs = colors[0]
		axs.plot(xs, PressureValueToPlotScenario1, linestyle='-', color=cs, label='pressure value no leak')
		cs = colors[1]
		axs.plot(xs, PressureValueToPlotScenario2, linestyle='--', color=cs, label='pressure value leak group 3(4)')
		cs = colors[2]
		axs.plot(xs, PressureValueToPlotScenario3, linestyle='-.', color=cs, label='pressure value leak group 5(4)')
		axs.set_xlabel('Nodes')
		axs.set_ylabel('[PSI]')
		axs.legend(ncol=2, loc='upper right')

		if (save):
			output_filename = "tensorflow_group_datasets/fig/node_network_group_slope_3.png"
			plt.savefig(output_filename, dpi=300, bbox_inches="tight")
			print("Saved to: " + output_filename)

		if (show):
			plt.show()

		plt.clf()


	return


	# cols = ["lost_0",  "lost_1", "lost_2",
	# 		"lost_3",  "lost_4", "lost_5",
	# 		"lost_6",  "lost_7", "lost_8",
	# 		"lost_9" ]
	#
	#
	# for ii in range(0,1):
	# 	print("FIGURE : ", ii)
	# 	# fig = plt.figure()
	# 	fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
	# 	# if ii<3:
	# 	# 	fig, axs = plt.subplots(4, 6, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
	# 	# else:
	# 	# 	fig, axs = plt.subplots(2, 6, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
	# 	# fig.suptitle('Sharing both axes')
	#
	# 	# nodeGroup = data[0:int(24*82)].groupby('nodeID').mean()
	# 	# nodeLeakageGroup = data_leakage[0:int(24*82)].groupby('nodeID').mean()
	#
	# 	# nodeGroup = data[0:int(24*82)].groupby('nodeID').mean()
	# 	# nodeLeakageGroup = data_leakage[0:int(24*82)].groupby('nodeID').mean()
	#
	#
	#
	# 	colorIndex = 0
	# 	for jj in range(0,80,10):
	# 		# print(jj)
	# 		node = nodeList[jj]
	# 		print("SUBPLOT node : ", node)
	# 		# ax1 = fig.add_subplot(4, 6, nodeIndex)
	#
	# 		# dataToPlot = data.loc[(data['nodeID'] == node)]
	#
	# 		#hour,nodeID,base_demand,demand_value,head_value,pressure_value,x_pos,y_pos,node_type,has_leak,leak_area_value,leak_discharge_value,current_leak_demand_value,smart_sensor_is_present,tot_network_demand
	# 		# baseDemandToPlot = dataToPlot['base_demand'].values #*  3.785412 * 60
	# 		# leakDemandToPlot = dataToPlot['leak_demand_value'].values #*  3.785412 * 60
	# 		# demandToPlot = dataToPlot['demand_value'].values #*  3.785412 * 60
	#
	# 		# lostDemandToPlot = dataToPlot.loc[jj, cols]
	#
	# 		lostDemandToPlot = nodeGroup.loc[node, cols]
	# 		leakageLostDemandPlot = nodeLeakageGroup.loc[node, cols]
	# 		leakage2LostDemandPlot = nodeLeakageGroup2.loc[node, cols]
	#
	# 		# for c, k in zip(colors, yticks):
	# 		cs = colors[colorIndex]
	# 		colorIndex += 1
	# 		# k = yticks[nodeIndex];
	#
	# 		# create data for the y=k 'layer'.
	# 		xs = np.arange(len(lostDemandToPlot))
	#
	#
	# 		# Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
	# 		# ax1.bar(xs, ys, k, zdir='y', color=cs, alpha=0.8)
	# 		# ax1.plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
	# 		# ax1.plot(xs, demandToPlot, label='Demand value node ' + str(node))
	#
	# 		# print(plotIndexRow, ' : ', plotIndexColumn)
	#
	# 		# axs[plotIndexRow, plotIndexColumn].plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
	# 		# axs[plotIndexRow, plotIndexColumn].plot(xs, demandToPlot, label='Demand value node ' + str(node))
	#
	# 		# axs[plotIndexRow, plotIndexColumn].plot(xs, baseDemandToPlot, label='Base Demand node')
	# 		# axs[plotIndexRow, plotIndexColumn].plot(xs, demandToPlot, label='Demand value node ')
	# 		# axs[plotIndexRow, plotIndexColumn].plot(xs, leakDemandToPlot, label='Leak Demand value node ')
	# 		#
	# 		# axs[plotIndexRow, plotIndexColumn].plot(xs, lostDemandToPlot, label='Lost Demand group ')
	# 		axs.plot(xs, lostDemandToPlot, linestyle='-', color=cs, label='Lost D G '+str(jj))
	# 		axs.plot(xs, leakageLostDemandPlot, linestyle='--', color=cs, label='Lost leak D G '+str(jj))
	# 		axs.plot(xs, leakage2LostDemandPlot, linestyle='-.', color=cs, label='Lost leak D G '+str(jj))
	#
	# 		axs.set_xlabel('Nodes')
	# 		axs.set_ylabel('[G/M]')
	#


	cols = ["pressure_0",  "pressure_1", "pressure_2",
			"pressure_3",  "pressure_4", "pressure_5",
			"pressure_6",  "pressure_7", "pressure_8",
			"pressure_9" ]

	for ii in range(0, 1):
		print("FIGURE : ", ii)
		# fig = plt.figure()
		fig5, axs5 = plt.subplots(1, 1, sharex=True, sharey=True, gridspec_kw={'hspace': 0})

		colorIndex = 0
		for jj in range(0, 80, 10):
			# print(jj)
			node = nodeList[jj]
			print("SUBPLOT node : ", node)

			lostDemandToPlot = nodeGroup.loc[node, cols]
			leakageLostDemandPlot = nodeLeakageGroup.loc[node, cols]
			leakage2LostDemandPlot = nodeLeakageGroup2.loc[node, cols]

			cs = colors[colorIndex]
			colorIndex += 1

			xs = np.arange(len(lostDemandToPlot))

			axs5.plot(xs, lostDemandToPlot, linestyle='-', color=cs, label='Lost D G ' + str(jj))
			axs5.plot(xs, leakageLostDemandPlot, linestyle='--', color=cs, label='Lost leak D G ' + str(jj))
			axs5.plot(xs, leakage2LostDemandPlot, linestyle='-.', color=cs, label='Lost leak D G ' + str(jj))

			axs5.set_xlabel('Nodes')
			axs5.set_ylabel('[G/M]')


		# axs.legend()

		# if ii == 1:
		#     ax1.set_zlabel('Base demand [G/M]')
		# else:
		#     ax1.set_zlabel('Demand value [G/M]')

		# On the y axis let's only label the discrete values that we have data for.
		# ax.set_yticks(yticks)
		# ax1.set_yticks([1,2,3,4])

		# bool_base_demand = True
		# bool_pressure_value = False
		# data = data.loc[(data['base_demand'] == bool_base_demand) & (data['pressure_value'] == bool_pressure_value)]
		#
		# # x = ["no leak", "1 leak", "1/8 leaks", "1/4 leaks", "1/2 leaks"]
		# # y = data[y_data]
		#
		# time_period = path[0:2]
		#
		# plot_title = time_period + " with "

		# output_filename = "one_res_small_no_leaks_node_demand_base_value_" + str(ii) + "_" #+ y_data + "_with_"
		# output_filename = path.replace("/", "-") + str(ii)

		# # output_filename = "1W_delta_loss_with_bdem_press.png"
		#
		# if (bool_base_demand == True and bool_pressure_value == False):
		#     plot_title = plot_title + "base_demand"
		#     output_filename = output_filename + "base_demand"
		# elif (bool_base_demand == False and bool_pressure_value == True):
		#     plot_title = plot_title + "pressure"
		#     output_filename = output_filename + "pressure"
		# elif (bool_base_demand == True and bool_pressure_value == True):
		#     plot_title = plot_title + "base_demand and pressure"
		#     output_filename = output_filename + "base_demand_pressure"
		#

		#output_filename = "fig2/" + output_filename + ".png"

		#
		# # plt.yticks(np.arange(0, max(y), 0.00001))
		#
		# # fig = plt.figure()
		# # ax = fig.add_subplot()
		#
		# # plt.plot(x, y)
		# plt.title(plot_title)
		#
		# # plt.plot(x,y,'ro')
		#
		# # for i,j in zip(x,y):
		# #     ax.annotate(str(j), xy=(i,j), xytext=(10,20), textcoords='offset points')
		#
		# plt.xlabel('Dataset')
		# # plt.ylabel('Delta Loss')
		#
		# # plt.ylabel(y_data)
		# plt.ylabel("user demand")
		#
		# # plt.legend()
		# plt.grid(True)
		#
		if (save):
			plt.savefig(output_filename, dpi=300, bbox_inches="tight")
			print("Saved to: " + output_filename)

		if (show):
			plt.show()

		plt.clf()



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

def plot_model_figure():
	import os
	os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

	import tensorflow as tf
	import pydot
	import pydotplus
	import graphviz

	leak_area = "0164"  # "0246" #"0164"
	# 1-->7 2-->7 3-->6 4-->2 5-->1 6-->7 7-->7
	leak_group = 3
	leak_group_model = 7

	dataset_path = "tensorflow_group_datasets/model/h5/"
	model_filename = "model_leak_group" + str(leak_group) + "_train_node_" + str(leak_group_model) + ".h5"
	loaded_model = tf.keras.models.load_model(dataset_path + model_filename)

	tf.keras.utils.plot_model(loaded_model, to_file='model_plot.png', show_shapes=True)


if __name__ == "__main__":

	# exported_path = 'tensorflow_group_datasets/fig/'
	# path_base = exported_path + "report_model_comparison_leak_group_"
	# plot_model_analysis(path_base, save=True, show=False)


	# plot_model_figure()
	# sys.exit(1)


	folder_input = "tensorflow_group_datasets/"

	### 1M no leak
	folder_network = "one_res_small/0_no_leaks_rand_base_demand/"
	input_full_dataset = folder_network + '1M_one_res_small_leaks_ordered_group_0_node_0_0164_merged.csv'
	complete_path = folder_input + input_full_dataset

	### 1M leak
	folder_network_leakage = "one_res_small/1_at_82_leaks_rand_base_demand/"
	input_full_dataset_leakage = folder_network_leakage + '1M_one_res_small_leaks_ordered_group_3_node_4_0164_merged.csv'
	complete_path_leakage = folder_input + input_full_dataset_leakage

	### 1M leak
	folder_network_leakage_2 = "one_res_small/1_at_82_leaks_rand_base_demand/"
	input_full_dataset_leakage_2 = folder_network_leakage_2 + '1M_one_res_small_leaks_ordered_group_5_node_4_0164_merged.csv'
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

	plot_lost_demand(train_dataset, train_dataset_leakage, train_dataset_leakage_2, True, True)
	sys.exit(1)








	# # path = "1D_tensorflow_report_2022-12-07_15_33_02_555157.csv"
	#
	# path = "1W_tensorflow_report_2022-12-06_19_01_27_721593.csv"
	# y_data = 'delta_loss'
	#
	# save = True
	# show = False
	#
	# #################
	#
	# bool_base_demand = True
	# bool_pressure_value = False
	#
	# delta_plot_after_tensorflow_analysis(path, bool_base_demand, bool_pressure_value, y_data, save=save, show=show)
	#
	# bool_base_demand = False
	# bool_pressure_value = True
	#
	# delta_plot_after_tensorflow_analysis(path, bool_base_demand, bool_pressure_value, y_data, save=save, show=show)
	#
	# bool_base_demand = True
	# bool_pressure_value = True
	#
	# delta_plot_after_tensorflow_analysis(path, bool_base_demand, bool_pressure_value, y_data, save=save, show=show)
	#
	# #################
	#
	# dataset = "1W_one_res_small_no_leaks_rand_base_dem_nodes_output.csv"
	#
	# y_data = 'epochs'
	#
	# simple_features_combination_plot_after_tensorflow_analysis(path, dataset, y_data, save=save, show=show)
	#
	# y_data = 'loss'
	#
	# simple_features_combination_plot_after_tensorflow_analysis(path, dataset, y_data, save=save, show=show)

	"""
	- ci sono demand_value predette negative, dovremmo mettere una regola che questo non è possibile
	- ci piace questo plot: https://matplotlib.org/stable/gallery/mplot3d/bars3d.html#sphx-glr-gallery-mplot3d-bars3d-py
	- ci piace anche questo: https://matplotlib.org/stable/gallery/mplot3d/polys3d.html#sphx-glr-gallery-mplot3d-polys3d-py forse meglio, 
	- PRIMO TIPO DI GRAFICO: 9 nodi per ogni grafico, tempo sulle x, nodo sul lambda, valore VERO sulle 'probability'
	- nel secondo tipo di grafico vogliamo plottare l'errore, cambiando il plot facendo vedere una superficie: https://matplotlib.org/stable/gallery/mplot3d/custom_shaded_3d_surface.html#sphx-glr-gallery-mplot3d-custom-shaded-3d-surface-py
	- SECONDO TIPO DI GRAFICO: 9 nodi per ogni grafico, tempo sulle x, nodo sul lambda, valore delta delle predictions-valore vero su 'probability'
	"""

	# path_rand_demand = "tensorflow_datasets/one_res_small/no_leaks_rand_base_demand/1W/1W_one_res_small_no_leaks_rand_base_dem_nodes_output.csv"

	# path_rand_demand = "tensorflow_datasets/one_res_small/no_leaks_rand_base_demand/1M/1M_one_res_small_no_leaks_rand_bd_merged.csv"
	# path_rand_demand = "tensorflow_datasets/one_res_small/1_at_2_leaks_rand_base_demand/1M/1M_one_res_small_leaks_rand_bd_merged.csv"

	# path_rand_demand = "tensorflow_datasets/one_res_small/1_at_2_leaks_rand_base_demand/1M/1M_one_res_small_leaks_rand_bd_a0001_merged.csv"
	# path_rand_demand = "tensorflow_datasets/one_res_small/1_at_2_leaks_rand_base_demand/1M/1M_one_res_small_leaks_rand_bd_a001_merged.csv"
	# path_rand_demand = "tensorflow_datasets/one_res_small/1_at_2_leaks_rand_base_demand/1M/1M_one_res_small_leaks_rand_bd_a0005_merged.csv"

	# # plot_user_demand(path_rand_demand, save=False, show=True)
	# plot_user_demand_v2(path_rand_demand, save=False, show=True)
	# # plot_hist_user_demand(path_rand_demand, save=True, show=False)
	# # plot_cumulative_hist_user_demand(path_rand_demand, save=True, show=False)

	# path_pattern_demand = "tensorflow_datasets/one_res_small/no_leaks_pattern_demand/1M/1M_one_res_small_alt_no_leaks_nodes_output.csv"
	# plot_user_demand(path_pattern_demand, save=False, show=True)
	# plot_user_demand_v2(path_pattern_demand, save=True, show=False)
	# # plot_hist_user_demand(path_pattern_demand, save=True, show=False)
	# plot_cumulative_hist_user_demand(path_pattern_demand, save=True, show=False)