import wntr
import pandas as pd
import csv

inp_file = 'networks/Net3.inp'

wn = wntr.network.WaterNetworkModel(inp_file)

node_names = wn.node_name_list

results = wntr.sim.EpanetSimulator(wn).run_sim()

demand_results= results.node['demand']
head_results= results.node['head']
pressure_results= results.node['pressure']

indexes = demand_results.index

outName = "nodes_output.csv"
out = open(outName, "w")
writer = csv.writer(out)

for timestamp in indexes:
    tot_demand = 0.0

    for nodeID in node_names:
        node_obj = wn.get_node(nodeID)

        demand_value = demand_results.loc[timestamp, nodeID]
        head_value = head_results.loc[timestamp, nodeID]
        pressure_value = pressure_results.loc[timestamp, nodeID]

        hour = pd.to_datetime(timestamp, unit='s').time()
        tot_demand += demand_value

        x_pos = node_obj.coordinates[0]
        y_pos = node_obj.coordinates[1]

        node_type = node_obj.__class__.__name__

        output_row = [hour, nodeID, demand_value, head_value, pressure_value, x_pos, y_pos, node_type]

        writer.writerow(output_row)

out.close()