import wntr
import csv
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime

def formatted_datetime():
    # current date and time
    now = str(datetime.now())
    now = now.replace(" ", "_")
    now = now.replace(".", "_")
    now = now.replace(":", "_")

    return now

def pick_rand_leaks(wn, number_of_junctions_with_leaks):
    node_names = wn.junction_name_list

    selected_junctions = random.sample(node_names, number_of_junctions_with_leaks)

    return selected_junctions

def assign_leaks(wn, area_size, selected_junctions):
    for node_id in selected_junctions:
        node_obj = wn.get_node(node_id)

        node_obj.add_leak(wn, area=area_size, start_time=0)

def create_custom_pattern(wn, name, min_mult, max_mult, step, duration):
    timeops = wntr.network.options.TimeOptions(duration)

    multipliers = []

    for multiplier_step in range(min_mult,max_mult+step,step):
        multipliers.append(multiplier_step)

    out_pattern = wntr.network.Pattern("custom", multipliers, time_options=timeops)

    wn.add_pattern(name, out_pattern)

    return out_pattern

def assign_rand_demand_to_junctions(wn, min_bd, max_bd, pattern=None, list_of_demands=None):
    node_names = wn.junction_name_list

    for juncID in node_names:
        junc_obj = wn.get_node(juncID)

        new_demand = random.uniform(min_bd, max_bd)

        # junc_obj.demand_timeseries_list[0].base_value = new_demand

        # you can't apparently just change the old pattern like the base demand, it seems to be a read only field.
        # for now we just create a new timeseries and delete the old one
        # junc_obj.demand_timeseries_list[0].pattern = pattern

        junc_obj.add_demand(base=new_demand, pattern_name=pattern)
        del junc_obj.demand_timeseries_list[0]

        # To keep track of the history of the different random base demands we add a custom field to the Junction object of wntr
        # if it is the first time that we call this, we have to create it, else we just append the new value to the list
        if hasattr(junc_obj, 'list_of_bds'):
            junc_obj.list_of_bds.append(junc_obj.base_demand)
        else:
            junc_obj.list_of_bds = [junc_obj.base_demand]

def write_results_to_csv(results, sim_duration, wn, out_filename, number_of_nodes_with_leaks, file_timestamp=False):
    print("Printing Nodes CSV. Please wait...")

    node_names = wn.node_name_list

    demand_results = results.node['demand']
    head_results = results.node['head']
    pressure_results = results.node['pressure']
    leak_demand_results = results.node["leak_demand"]

    sim_duration_in_hours = int(sim_duration / 3600)

    now = formatted_datetime()

    if(file_timestamp):
        out_filename_complete = out_filename + "_nodes_output_"+now+".csv"
    else:
        out_filename_complete = out_filename + "_nodes_output.csv"

    out = open(out_filename_complete, "w", newline='', encoding='utf-8')
    writer = csv.writer(out)

    header = ["hour", "nodeID", "base_demand", "demand_value", "head_value",
              "pressure_value", "x_pos", "y_pos", "node_type", "has_leak",
              "leak_area_value", "leak_discharge_value",
              "leak_demand_value",
              "tot_junctions_demand", "tot_leaks_demand","tot_network_demand"]

    writer.writerow(header)

    # These two variables are needed for the simulation stats
    tot_juncts_demand_in_entire_simulation = np.float64(0)
    tot_juncts_leak_demand_in_entire_simulation = np.float64(0)

    for timestamp in range(sim_duration_in_hours):

        # Water network balancing counters reset each hour
        tot_leaks_demand = np.float64(0)
        tot_junctions_demand = np.float64(0)
        tot_network_demand = np.float64(0)

        for nodeID in node_names:
            node_obj = wn.get_node(nodeID)
            node_type = node_obj.__class__.__name__

            hour_in_seconds = int(timestamp * 3600)

            hour = str(timestamp) + ":00:00"

            demand_value = demand_results.loc[hour_in_seconds,nodeID]
            tot_junctions_demand += demand_value

            head_value = head_results.loc[hour_in_seconds, nodeID]
            head_value = "{:.8f}".format(head_value)

            pressure_value = pressure_results.loc[hour_in_seconds,nodeID]
            pressure_value = "{:.8f}".format(pressure_value)

            x_pos = node_obj.coordinates[0]
            y_pos = node_obj.coordinates[1]

            leak_area_value = node_obj.leak_area  # I think that this does not require an approximation... right?
            leak_discharge_value = node_obj.leak_discharge_coeff

            leak_demand_value = leak_demand_results.loc[hour_in_seconds, nodeID]
            tot_leaks_demand += leak_demand_value

            if node_type == "Junction":

                if hasattr(node_obj, 'list_of_bds'):
                    #happens if we set random_base_demands to True
                    base_demand = node_obj.list_of_bds[timestamp]
                else:
                    base_demand = node_obj.demand_timeseries_list[0].base_value

                base_demand = "{:.8f}".format(base_demand)

                tot_juncts_demand_in_entire_simulation += demand_value
                tot_juncts_leak_demand_in_entire_simulation += leak_demand_value
            else:
                base_demand = 0.0

            tot_network_demand += demand_value + leak_demand_value

            tot_network_demand_str = "{:.8f}".format(tot_network_demand)
            leak_demand_value = "{:.8f}".format(leak_demand_value)
            demand_value = "{:.8f}".format(demand_value)

            if (leak_area_value > 0.0):
                has_leak = True  # this leak-flag is set to true if we see a hole in the node
            else:
                has_leak = False

            tot_junctions_demand_str = "{:.8f}".format(tot_junctions_demand)
            tot_leaks_demand_str = "{:.8f}".format(tot_leaks_demand)

            out_row = [hour,nodeID,base_demand, demand_value, head_value, pressure_value,
                       x_pos, y_pos, node_type, has_leak, leak_area_value,
                       leak_discharge_value, leak_demand_value,
                       tot_junctions_demand_str, tot_leaks_demand_str, tot_network_demand_str]

            writer.writerow(out_row)

    out.close()
    print("CSV saved to: "+out_filename_complete+"\n")

    stats_filename = write_simulation_stats(wn, out_filename, tot_juncts_demand_in_entire_simulation, tot_juncts_leak_demand_in_entire_simulation, number_of_nodes_with_leaks, now=now)

    return out_filename_complete, stats_filename

def write_simulation_stats(wn, out_file_name, tot_nodes_demand, tot_leak_demand, number_of_nodes_with_leaks, now=None):
    print("Writing simulation stats CSV...")

    if (now == None):
        outName = out_file_name + "_nodes_simulation_stats.csv"
    else:
        outName = out_file_name + "_nodes_simulation_stats_" + now + ".csv"

    out = open(outName, "w", newline='', encoding='utf-8')
    writer = csv.writer(out)

    header = ["tot_nodes_demand", "leak_percentage", "number_of_nodes", "number_of_junctions",
              "number_of_reservoirs", "number_of_tanks", "number_of_nodes_with_leaks",
              "time_spent_on_sim"]

    writer.writerow(header)

    number_of_nodes = len(wn.node_name_list)
    number_of_junctions = len(wn.junction_name_list)
    number_of_reservoirs = len(wn.reservoir_name_list)
    number_of_tanks = len(wn.tank_name_list)
    time_spent_on_sim = int(((wn.options.time.duration) / 3600)) + 1 #see in run_sim why we do +1

    if (tot_nodes_demand > 0):
        leak_percentage = (tot_leak_demand / tot_nodes_demand) * 100
        leak_percentage = round(leak_percentage, 4)
    else:
        leak_percentage = 0.0

    tot_nodes_demand = "{:.8f}".format(tot_nodes_demand)
    tot_leak_demand = "{:.8f}".format(tot_leak_demand)

    print("\nTot demand for Nodes only is: " + str(tot_nodes_demand) + " and tot_leak_demand is: " + str(
        tot_leak_demand))
    print("Total leak demand for nodes is:  " + str(leak_percentage) + "% of the Total Nodes' demand")
    print("Number of nodes inside of the network is: " + str(number_of_nodes))
    print("Number of Junctions only: " + str(number_of_junctions))
    print("Number of Reservoirs only: " + str(number_of_reservoirs))
    print("Number of Tanks only: " + str(number_of_tanks))
    print("Number of Junctions with leaks: " + str(number_of_nodes_with_leaks))
    print("Total hours simulated: " + str(time_spent_on_sim) + " (i.e. from 0:00:00 to "+str(int(time_spent_on_sim-1))+":00:00)\n")

    output_row = [tot_nodes_demand, leak_percentage, number_of_nodes, number_of_junctions,
                  number_of_reservoirs, number_of_tanks, number_of_nodes_with_leaks,
                  time_spent_on_sim]

    writer.writerow(output_row)

    out.close()

    print("Simulation stats saved to: "+outName+"\n")

    return outName

def run_sim(sim_folder_path, input_file_inp, sim_duration, out_filename, leaks_enabled=False, leak_area_size=0.0000001, random_base_demands=False, min_bd=0, max_bd=0.000005, file_timestamp=False):
    print("Configuring simulation...")

    complete_input_path = sim_folder_path + input_file_inp

    print("Loading INP file at: "+complete_input_path)

    wn = wntr.network.WaterNetworkModel(complete_input_path)

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    sim_duration_for_wntr = sim_duration - 3600

    # Why -3600? WNTR adds an hour to the sim! e.g. if we set 24 hrs it will simulate from 0:00 to 24:00 (included), so 25 hrs in total
    wn.options.time.duration = sim_duration_for_wntr

    # wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    # wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    print("Demand mode: "+str(wn.options.hydraulic.demand_model))
    print("Required pressure: "+str(wn.options.hydraulic.required_pressure))
    print("Minimum pressure: "+str(wn.options.hydraulic.minimum_pressure))
    print("Time duration (seconds): "+str(sim_duration))
    print("WNTR duration (seconds): "+str(sim_duration_for_wntr))

    if(leaks_enabled):
        print("LEAKS ARE ENABLED")

        number_of_junctions_with_leaks = int(len(wn.junction_name_list) / 2)

        selected_junctions = pick_rand_leaks(wn, number_of_junctions_with_leaks)

        assign_leaks(wn, leak_area_size, selected_junctions)
    else:
        number_of_junctions_with_leaks = 0
        print("Leaks are NOT enabled")

    if(random_base_demands):
        print("RANDOM BASE DEMANDS ENABLED")

        results_list = execute_simulation_with_random_base_demands(wn, sim_duration_for_wntr, min_bd=min_bd, max_bd=max_bd)

        results = make_a_single_results_from_the_list(wn, results_list)
    else:
        print("Random Base Demands are NOT enabled")

        results = execute_simulation(wn)

    saved_datasets = write_results_to_csv(results, sim_duration, wn, out_filename, number_of_junctions_with_leaks, file_timestamp=file_timestamp)

    print("Simulation finished")

    return saved_datasets

def execute_simulation(wn):
    print("\nRunning simulation...")

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    return results

def make_a_single_results_from_the_list(wn, results_list):

    # fake temp class to give to the write_csv function
    class Results:
        def __init__(self, node={}):
            self.node = node

    columns = wn.node_name_list

    node = OrderedDict({'demand': pd.DataFrame(columns=columns), 'head': pd.DataFrame(columns=columns), 'pressure': pd.DataFrame(columns=columns), 'leak_demand': pd.DataFrame(columns=columns)})

    for results in results_list:

        demand_results = results.node['demand']
        head_results = results.node['head']
        pressure_results = results.node['pressure']
        leak_demand_results = results.node["leak_demand"]

        # merge the two dataframes vertically (row-wise) using pd.concat
        node['demand'] = pd.concat([node['demand'], demand_results], axis=0)
        node['head'] = pd.concat([node['head'], head_results], axis=0)
        node['pressure'] = pd.concat([node['pressure'], pressure_results], axis=0)
        node['leak_demand'] = pd.concat([node['leak_demand'], leak_demand_results], axis=0)

    out = Results(node)

    return out

def execute_simulation_with_random_base_demands(wn, sim_duration_for_wntr, min_bd=0, max_bd=0.000005):
    print("\nRunning simulation...")

    pattern = create_custom_pattern(wn,"custom_1",1,1,1,sim_duration_for_wntr)

    sim_duration_in_hours = int(sim_duration_for_wntr / 3600) + 1

    results_list = []

    for hour in range(sim_duration_in_hours):

        wn.options.time.duration = hour * 3600

        assign_rand_demand_to_junctions(wn, min_bd, max_bd, "custom_1")

        results = wntr.sim.WNTRSimulator(wn).run_sim()

        results_list.append(results)

    return results_list

def merge_multiple_datasets(datasets_to_merge, output_filename):
    print("Merging these datasets:")

    pd.options.display.float_format = '{:,.8f}'.format

    path_to_first_df = datasets_to_merge.pop(0)
    print(path_to_first_df)

    # We read our entire dataset
    first_df = pd.read_csv(path_to_first_df)

    last_row_from_first_df = first_df.iloc[-1]["hour"]

    # cheap hack useful to know how many nodes we have in a dataset
    number_of_nodes = first_df['hour'].value_counts()[last_row_from_first_df]

    last_hour_from_first_df = int(last_row_from_first_df.split(":",1)[0]) + 1

    while len(datasets_to_merge) > 0:
        data_path = datasets_to_merge.pop(0)
        print(path_to_first_df)

        next_df = pd.read_csv(data_path, header=0)

        n_iterations = int(len(next_df) / number_of_nodes)

        for mult in range(1, n_iterations+1):
            stop = mult * number_of_nodes
            start = stop - number_of_nodes
            value = str(last_hour_from_first_df)+":00:00"
            next_df.loc[start:stop, 'hour'] = value

            last_hour_from_first_df += 1

        first_df = pd.concat([first_df, next_df], ignore_index=True)

    first_df.to_csv(output_filename, float_format='%.8f', index=False)

    print("Merge finished. Final csv saved to: "+output_filename)

if __name__ == "__main__":
    print("******   py_epanet started!  ******\n")

    # input_file_inp = "Net3.inp"
    input_file_inp = "exported_month_large_complete_one_reservoirs_small.inp"
    sim_folder_path = "./networks/"
    out_filename = "1D_one_res_small_no_leaks"

    sim_duration = 24 * 3600  # hours in seconds

    leaks_enabled = False  # switch this to True to enable leaks assignments
    leak_area_size = 0.0000001  # area of the "hole" of the leak

    random_base_demands = True  # switch this to True to enable random base demand assignments
    min_bd = 0  # minimum possible random base demand
    max_bd = 0.000005  # maximum possible random base demand

    file_timestamp = True  # switch this to True to write a current timestamp to the output filename

    # SINGLE EXECUTION
    # run_sim(sim_folder_path, input_file_inp, sim_duration, out_filename,
    #         leaks_enabled=leaks_enabled, leak_area_size=leak_area_size,
    #         random_base_demands=random_base_demands, file_timestamp=file_timestamp)

    datasets_to_merge = []

    number_of_consecutive_sims = 31

    for i in range(number_of_consecutive_sims):
        results_from_sim = run_sim(sim_folder_path, input_file_inp, sim_duration,
                                   out_filename, leaks_enabled=leaks_enabled,
                                   leak_area_size=leak_area_size,
                                   random_base_demands=random_base_demands,
                                   min_bd=min_bd, max_bd=max_bd,file_timestamp=file_timestamp)

        datasets_to_merge.append(results_from_sim[0])

    print()

    merge_multiple_datasets(datasets_to_merge, "1D_one_res_small_no_leaks_nodes_output_merged.csv")

    print("\nExiting...")