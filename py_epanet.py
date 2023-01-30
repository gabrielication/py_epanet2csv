import wntr
import csv
import sys
import random
import numpy as np
import pandas as pd

def write_results_to_csv(results, node_names, sim_duration, wn, out_filename):
    print("Printing Nodes CSV. Please wait...")

    demand_results = results.node['demand']
    head_results = results.node['head']
    pressure_results = results.node['pressure']

    sim_duration_in_hours = int(sim_duration / 3600)

    out_filename_complete = out_filename+"_nodes_output.csv"

    out = open(out_filename_complete, "w", newline='', encoding='utf-8')
    writer = csv.writer(out)

    header = ["hour", "nodeID", "base_demand", "demand_value", "head_value",
              "pressure_value", "x_pos", "y_pos", "node_type", "has_leak",
              "leak_area_value", "leak_discharge_value",
              "current_leak_demand_value", "tot_network_demand"]

    writer.writerow(header)

    for timestamp in range(sim_duration_in_hours):
        for nodeID in node_names:
            node_obj = wn.get_node(nodeID)
            node_type = node_obj.__class__.__name__

            hour_in_seconds = int(timestamp * 3600)

            hour = str(timestamp) + ":00:00"
            demand_value = demand_results.loc[hour_in_seconds,nodeID]
            demand_value = "{:.8f}".format(demand_value)

            head_value = head_results.loc[hour_in_seconds, nodeID]
            head_value = "{:.8f}".format(head_value)

            pressure_value = pressure_results.loc[hour_in_seconds,nodeID]
            pressure_value = "{:.8f}".format(pressure_value)

            x_pos = node_obj.coordinates[0]
            y_pos = node_obj.coordinates[1]

            if node_type == "Junction":
                base_demand = node_obj.demand_timeseries_list[0].base_value
                base_demand = "{:.8f}".format(base_demand)
            else:
                base_demand = 0.0

            #TODO
            has_leak = False
            leak_area_value = 0.0
            leak_discharge_value = 0.0
            current_leak_demand_value = 0.0
            tot_network_demand = 0.0

            out_row = [hour,nodeID,base_demand, demand_value, head_value, pressure_value,
                       x_pos, y_pos, node_type, has_leak, leak_area_value,
                       leak_discharge_value, current_leak_demand_value, tot_network_demand]

            writer.writerow(out_row)

    print("CSV writing finished")
    out.close()
    print("CSV saved to: "+out_filename_complete)


def run_sim(sim_folder_path, input_file_inp, sim_duration, out_filename):
    print("Simulation started...")

    complete_input_path = sim_folder_path + input_file_inp

    print("Loading INP file at: "+complete_input_path)

    wn = wntr.network.WaterNetworkModel(complete_input_path)

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    # wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    # wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    wn.options.time.duration = sim_duration

    print("Running simulation...")

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    node_names = wn.node_name_list

    write_results_to_csv(results, node_names, sim_duration, wn, out_filename)

    print("Simulation finished")

if __name__ == "__main__":
    print("py_epanet started!\n")

    input_file_inp = "exported_month_large_complete_one_reservoirs_small.inp"
    sim_folder_path = "./networks/"

    sim_duration = 24 * 3600
    out_filename = "1D_one_res_small_no_leaks"

    run_sim(sim_folder_path, input_file_inp, sim_duration, out_filename)
    print("\nExiting...")