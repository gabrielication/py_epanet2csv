import argparse

import wntr
import pandas as pd
import csv
import sys
import numpy as np
from decimal import Decimal

def run_sim(inp_file):
    print("Running simulation...")

    wn = wntr.network.WaterNetworkModel(inp_file)

    node_names = wn.node_name_list
    link_names = wn.link_name_list

    results = wntr.sim.EpanetSimulator(wn).run_sim()

    print("Simulation finished. Writing to csv (can take a while)...")

    nodes_to_csv(wn, results, node_names)
    links_to_csv(wn, results, link_names)

    print("Finished!")

def links_to_csv(wn, results, link_names):
    print("Writing Links' CSV...")

    flow_results = results.link['flowrate']
    velocity_results = results.link['velocity']
    headloss_results = results.link['headloss']

    indexes = flow_results.index

    outName = "links_output.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    debug = False

    for timestamp in indexes:
        tot_demand = Decimal('0')
        demand_value = Decimal('0')

        hour = pd.to_datetime(timestamp, unit='s').time()

        #print(timestamp)
        for linkID in link_names:
            link_obj = wn.get_link(linkID)

            flow_value = Decimal(str(flow_results.loc[timestamp, linkID]))
            flow_value = round(flow_value, 8)

            velocity_value = Decimal(str(velocity_results.loc[timestamp, linkID]))
            velocity_value = round(velocity_value, 8)

            headloss_value = Decimal(str(headloss_results.loc[timestamp, linkID]))
            headloss_value = round(headloss_value, 8)

            start_node_value = link_obj.start_node_name
            end_node_value = link_obj.end_node_name

            node_type = link_obj.__class__.__name__

            output_row = [hour, linkID, flow_value, velocity_value, headloss_value, start_node_value, end_node_value, node_type]

            writer.writerow(output_row)

    out.close()

    print("Links' CSV written...")

def nodes_to_csv(wn, results, node_names):
    print("Writing Nodes' CSV...")

    demand_results = results.node['demand']
    head_results = results.node['head']
    pressure_results = results.node['pressure']

    indexes = demand_results.index

    outName = "nodes_output.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    debug = False

    for timestamp in indexes:
        tot_demand = Decimal('0')
        demand_value = Decimal('0')

        hour = pd.to_datetime(timestamp, unit='s').time()

        #print(timestamp)
        for nodeID in node_names:
            node_obj = wn.get_node(nodeID)

            demand_value = Decimal(str(demand_results.loc[timestamp, nodeID]))
            demand_value = round(demand_value,8)
            head_value = Decimal(str(head_results.loc[timestamp, nodeID]))
            head_value = round(head_value,8)
            pressure_value = Decimal(str(pressure_results.loc[timestamp, nodeID]))
            pressure_value = round(pressure_value,8)


            if debug:
                print("--------")
                print(nodeID)
                print("demand")
                print("{:4.15}".format(demand_value))
                print("tot_d")
                print("{:4.15f}".format(tot_demand))

            tot_demand = tot_demand + demand_value
            tot_demand = round(tot_demand,8)

            if debug:
                print("tot_d")
                print("{:4.15f}".format(tot_demand))

            if debug:
                if nodeID=="101":
                    print("test")
                    sys.exit(1)

            x_pos = node_obj.coordinates[0]
            y_pos = node_obj.coordinates[1]

            node_type = node_obj.__class__.__name__

            output_row = [hour, nodeID, demand_value, head_value, pressure_value, x_pos, y_pos, node_type]
            # print(nodeID)
            # print(demand_value)
            # print(tot_demand)
            # print("Node (" + str(nodeID) + ") demand is: "+str(demand_value)+" so Tot demand is now: " + str(tot_demand))
            # print("Node ({}) demand is: {} so Tot demand is now: {}".format(nodeID,demand_value,tot_demand))

            writer.writerow(output_row)

        # print(timestamp)
        if debug:
            if tot_demand<1e-6:
                print("Tot demand at "+str(hour)+" is: 0 ("+str(tot_demand)+")")
            else:
                print("Tot demand at "+str(hour)+" is: "+str(tot_demand))
        # break

    out.close()

    print("Nodes' CSV written...")

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

    # nodes_path = "./extracted_nodes.csv"
    # links_path = "./extracted_links.csv"
    # nodes_path = "./nodes_output.csv"

    input_file_inp = "./networks/Net3.inp"
    run_sim(input_file_inp)