import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations

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

if __name__ == "__main__":
    # path = "1D_tensorflow_report_2022-12-07_15_33_02_555157.csv"

    path = "1W_tensorflow_report_2022-12-06_19_01_27_721593.csv"
    y_data = 'delta_loss'

    save = True
    show = False

    #################

    bool_base_demand = True
    bool_pressure_value = False

    delta_plot_after_tensorflow_analysis(path, bool_base_demand, bool_pressure_value, y_data, save=save, show=show)

    bool_base_demand = False
    bool_pressure_value = True

    delta_plot_after_tensorflow_analysis(path, bool_base_demand, bool_pressure_value, y_data, save=save, show=show)

    bool_base_demand = True
    bool_pressure_value = True

    delta_plot_after_tensorflow_analysis(path, bool_base_demand, bool_pressure_value, y_data, save=save, show=show)

    #################

    dataset = "1W_one_res_small_no_leaks_rand_base_dem_nodes_output.csv"

    y_data = 'epochs'

    simple_features_combination_plot_after_tensorflow_analysis(path, dataset, y_data, save=save, show=show)

    y_data = 'loss'

    simple_features_combination_plot_after_tensorflow_analysis(path, dataset, y_data, save=save, show=show)