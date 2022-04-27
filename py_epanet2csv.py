import argparse

import wntr
import pandas as pd
import csv

def run_nodes_sim_to_csv(inp_file):
    print("Running simulation...")

    wn = wntr.network.WaterNetworkModel(inp_file)

    node_names = wn.node_name_list

    results = wntr.sim.EpanetSimulator(wn).run_sim()

    print("Simulation finished. Writing to csv (can take a while)...")

    demand_results = results.node['demand']
    head_results = results.node['head']
    pressure_results = results.node['pressure']

    indexes = demand_results.index

    outName = "nodes_output.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    for timestamp in indexes:
        tot_demand = 0.0
        hour = pd.to_datetime(timestamp, unit='s').time()

        for nodeID in node_names:
            node_obj = wn.get_node(nodeID)

            demand_value = demand_results.loc[timestamp, nodeID]
            head_value = head_results.loc[timestamp, nodeID]
            pressure_value = pressure_results.loc[timestamp, nodeID]

            tot_demand += demand_value

            x_pos = node_obj.coordinates[0]
            y_pos = node_obj.coordinates[1]

            node_type = node_obj.__class__.__name__

            output_row = [hour, nodeID, demand_value, head_value, pressure_value, x_pos, y_pos, node_type]

            writer.writerow(output_row)

        print("Tot demand at "+str(hour)+" is: "+str(tot_demand))

    out.close()

    print("Finished!")

if __name__ == "__main__":
    # Program description
    text = "This program processes the report file from EPANET 2.2, extracts Node Results and converts them to csv. " \
           "Please type the path of the file you want to convert. "

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)

    # Add long and short argument
    parser.add_argument("--input", "-i", help="Input file for processing")

    # Read arguments from the command line
    args = parser.parse_args()

    # Check for --name
    if args.input:
        input_file = args.input
        run_nodes_sim_to_csv(input_file)
    else:
        print("See 'py_epanet2csv.py -h' in order to know how to correctly execute this program")