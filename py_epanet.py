import wntr
import csv
import sys
import random
import numpy as np
import pandas as pd
from decimal import Decimal

def pick_rand_leaks(wn, number_of_junctions_with_leaks):
    node_names = wn.junction_name_list

    selected_junctions = random.sample(node_names, number_of_junctions_with_leaks)

    return selected_junctions

def assign_leaks(wn, area_size, selected_junctions):
    for node_id in selected_junctions:
        node_obj = wn.get_node(node_id)

        node_obj.add_leak(wn, area=area_size, start_time=0)

def write_results_to_csv(results, node_names, sim_duration, wn, out_filename, number_of_nodes_with_leaks):
    print("Printing Nodes CSV. Please wait...")

    demand_results = results.node['demand']
    head_results = results.node['head']
    pressure_results = results.node['pressure']
    leak_demand_results = results.node["leak_demand"]

    sim_duration_in_hours = int(sim_duration / 3600)

    out_filename_complete = out_filename+"_nodes_output.csv"

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

    write_simulation_stats(wn, out_filename, tot_juncts_demand_in_entire_simulation, tot_juncts_leak_demand_in_entire_simulation, number_of_nodes_with_leaks)

def write_simulation_stats(wn, out_file_name, tot_nodes_demand, tot_leak_demand, number_of_nodes_with_leaks):
    print("Writing simulation stats CSV...")

    outName = out_file_name + "_nodes_simulation_stats.csv"
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
    time_spent_on_sim = ((wn.options.time.duration+1) / 3600) #see in run_sim why we do +1

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

def run_sim(sim_folder_path, input_file_inp, sim_duration, out_filename, leaks_enabled=False, leak_area_size=0.0000001):
    print("Configuring simulation...")

    complete_input_path = sim_folder_path + input_file_inp

    print("Loading INP file at: "+complete_input_path)

    wn = wntr.network.WaterNetworkModel(complete_input_path)

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    # Why -1? WNTR adds an hour to the sim! e.g. if we set 24 hrs it will simulate from 0:00 to 24:00 (included), so 25 hrs in total
    wn.options.time.duration = sim_duration - 1

    # wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    # wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    print("Demand mode: "+str(wn.options.hydraulic.demand_model))
    print("Required pressure: "+str(wn.options.hydraulic.required_pressure))
    print("Minimum pressure: "+str(wn.options.hydraulic.minimum_pressure))
    print("Time duration (seconds): "+str(sim_duration))

    if(leaks_enabled):
        print("LEAKS ARE ENABLED")

        number_of_junctions_with_leaks = int(len(wn.junction_name_list) / 2)

        selected_junctions = pick_rand_leaks(wn, number_of_junctions_with_leaks)

        assign_leaks(wn, leak_area_size, selected_junctions)
    else:
        number_of_junctions_with_leaks = 0
        print("Leaks are NOT enabled")

    print("\nRunning simulation...")

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    node_names = wn.node_name_list

    write_results_to_csv(results, node_names, sim_duration, wn, out_filename, number_of_junctions_with_leaks)

    print("Simulation finished")

if __name__ == "__main__":
    print("******   py_epanet started!  ******\n")

    input_file_inp = "exported_month_large_complete_one_reservoirs_small.inp"
    sim_folder_path = "./networks/"

    sim_duration = 24 * 3600
    out_filename = "1D_one_res_small_no_leaks"

    leaks_enabled = True
    leak_area_size = 0.0000009

    run_sim(sim_folder_path, input_file_inp, sim_duration, out_filename, leaks_enabled=leaks_enabled, leak_area_size=leak_area_size)

    print("\nExiting...")