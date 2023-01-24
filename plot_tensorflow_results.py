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
            #demandToPlot = dataToPlot['demand_value'].values #*  3.785412 * 60

            # for c, k in zip(colors, yticks):
            # c = colors[nodeIndex]; k = yticks[nodeIndex];

            # create data for the y=k 'layer'.
            xs = np.arange(len(baseDemandToPlot))

            # You can provide either a single color or an array with the same length as
            # xs and ys. To demonstrate this, we color the first bar of each set cyan.
            # cs = [c] * len(xs)
            # cs[0] = 'c'

            # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
            # ax1.bar(xs, ys, k, zdir='y', color=cs, alpha=0.8)
            # ax1.plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
            # ax1.plot(xs, demandToPlot, label='Demand value node ' + str(node))

            # axs[plotIndexRow, plotIndexColumn].plot(xs, baseDemandToPlot, label='Base Demand node ' + str(node))
            # axs[plotIndexRow, plotIndexColumn].plot(xs, demandToPlot, label='Demand value node ' + str(node))
            axs[plotIndexRow, plotIndexColumn].plot(xs, baseDemandToPlot, label='Base Demand node')
            # axs[plotIndexRow, plotIndexColumn].plot(xs, demandToPlot, label='Demand value node ')

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

        # demandToPlot = dataToPlot['demand_value'].values
        baseDemandToPlot = dataToPlot['base_demand'].values

        count, division = np.histogram(baseDemandToPlot, bins=40, range=[0, 0.01], density=False)
        # count, division = np.histogram(baseDemandToPlot, bins=20, density=False)
        #count, division = np.histogram(baseDemandToPlot, bins=20)


        print(count)
        print(division)

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
        if nodeIndex > 3: #81:
            break

    # ax.set_xticklabels(center.round(6), rotation=65)

    ax.set_xlabel('Demand [center bin (20)]')
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
    output_filename = "one_res_small_no_leaks_cdf_node_demand" + "_" #+ y_data + "_with_"
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




if __name__ == "__main__":
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
    path_rand_demand = "tensorflow_datasets/one_res_small/no_leaks_rand_base_demand/1W/1W_one_res_small_no_leaks_rand_base_dem_nodes_output.csv"
    # # plot_user_demand(path_rand_demand, save=False, show=True)
    # plot_user_demand_v2(path_rand_demand, save=False, show=True)
    # # plot_hist_user_demand(path_rand_demand, save=True, show=False)
    # # plot_cumulative_hist_user_demand(path_rand_demand, save=True, show=False)

    path_pattern_demand = "tensorflow_datasets/one_res_small/no_leaks_pattern_demand/1M/1M_one_res_large_nodes_output.csv"
    # plot_user_demand(path_pattern_demand, save=False, show=True)
    #plot_user_demand_v2(path_pattern_demand, save=True, show=False)
    # # plot_hist_user_demand(path_pattern_demand, save=True, show=False)
    plot_cumulative_hist_user_demand(path_pattern_demand, save=False, show=True)