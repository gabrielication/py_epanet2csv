import pandas as pd
import warnings
import numpy as np
import ast

from sklearn.preprocessing import OneHotEncoder

def process_dataset_for_binary_classification(input_filename, output_filename):
    df = pd.read_csv(input_filename)

    columns = df[df["hour"] == "0:00:00"]["nodeID"].array

    encoder = OneHotEncoder(sparse=False)

    encoded = encoder.fit_transform(np.array(columns).reshape(-1, 1))

    max_hour = int(df.iloc[-1]["hour"].split(":")[0])

    out_dict = {}

    for hour in range(max_hour):
        timestamp = str(hour) + ":00:00"

        print("Processing timestamp: ", timestamp)

        leaks_temp = []

        node_id_count = 0

        for nodeID in columns:
            # print(timestamp, nodeID)

            temp = df[df["hour"] == timestamp]
            temp = temp[temp["nodeID"] == nodeID]

            # node_id = float(temp.iloc[-1]["nodeID"])
            node_id = encoded[node_id_count]
            node_id_count += 1

            node_type = temp.iloc[-1]["node_type"]

            if node_type == "Junction":
                node_type = 0.0
            else:
                node_type = 1.0

            base_demand = float(temp.iloc[-1]["base_demand"])
            demand_value = float(temp.iloc[-1]["demand_value"])
            head_value = float(temp.iloc[-1]["head_value"])
            pressure_value = float(temp.iloc[-1]["pressure_value"])
            x_pos = float(temp.iloc[-1]["x_pos"])
            y_pos = float(temp.iloc[-1]["y_pos"])
            # end_node_link = float(temp.iloc[-1]["end_node_link"])

            has_leak = int(temp.iloc[-1]["has_leak"])
            leaks_temp.append(has_leak)

            output = [node_type, base_demand, demand_value, head_value, pressure_value, x_pos, y_pos]

            output.extend(node_id)

            # row.extend(output)

            if nodeID not in out_dict:
                out_dict[nodeID] = [output]
            else:
                out_dict[nodeID].append(output)

        if "has_leak" not in out_dict:
            out_dict["has_leak"] = [leaks_temp]
        else:
            out_dict["has_leak"].append(leaks_temp)

        print()

    out_df = pd.DataFrame(out_dict)
    # out_df.to_csv("conv1d_transposed_dataset.csv", index=False)

    out_df.to_pickle(output_filename+'.pickle')

def process_dataset_for_regression(input_filename, output_filename):
    df = pd.read_csv(input_filename)

    columns = df[df["hour"] == "0:00:00"]["nodeID"].array

    encoder = OneHotEncoder(sparse=False)

    encoded = encoder.fit_transform(np.array(columns).reshape(-1, 1))

    max_hour = int(df.iloc[-1]["hour"].split(":")[0])

    out_dict = {}

    for hour in range(max_hour):
        timestamp = str(hour) + ":00:00"

        print("Processing timestamp: ", timestamp)

        demands_temp = []

        node_id_count = 0

        for nodeID in columns:
            # print(timestamp, nodeID)

            temp = df[df["hour"] == timestamp]
            temp = temp[temp["nodeID"] == nodeID]

            # node_id = float(temp.iloc[-1]["nodeID"])
            node_id = encoded[node_id_count]
            node_id_count += 1

            node_type = temp.iloc[-1]["node_type"]

            if node_type == "Junction":
                node_type = 0.0
            else:
                node_type = 1.0

            base_demand = float(temp.iloc[-1]["base_demand"]) * 1000
            head_value = float(temp.iloc[-1]["head_value"]) * 1000
            pressure_value = float(temp.iloc[-1]["pressure_value"]) * 1000
            x_pos = float(temp.iloc[-1]["x_pos"])
            y_pos = float(temp.iloc[-1]["y_pos"])

            demand_value = float(temp.iloc[-1]["demand_value"]) * 1000
            demands_temp.append(demand_value)

            output = [node_type, base_demand, head_value, pressure_value, x_pos, y_pos]

            output.extend(node_id)

            # row.extend(output)

            if nodeID not in out_dict:
                out_dict[nodeID] = [output]
            else:
                out_dict[nodeID].append(output)

        if "demand_value" not in out_dict:
            out_dict["demand_value"] = [demands_temp]
        else:
            out_dict["demand_value"].append(demands_temp)

        print()

    out_df = pd.DataFrame(out_dict)
    # out_df.to_csv("conv1d_transposed_dataset.csv", index=False)

    out_df.to_pickle(output_filename+'.pickle')

def create_group(nodeID, group_id, groups_dict, nodes_dict):
    group = [nodeID]
    groups_dict[group_id] = group
    nodes_dict[nodeID] = group_id

def extract_subgroups_from_epanet_network(input_filename, max_number_of_nodes_in_group):
    df = pd.read_csv(input_filename)

    nodes_df = df[df["hour"] == "0:00:00"][["nodeID","x_pos","y_pos", "start_node_link", "end_node_link"]]

    sorted_coordinates_df = nodes_df.sort_values(by=["x_pos","y_pos"])

    set_of_node_ids = sorted_coordinates_df["nodeID"]

    # Convert array elements to dictionary keys
    nodes_dict = {elem: -1 for elem in set_of_node_ids}

    groups_dict = {}

    group_id = 0

    for nodeID in nodes_dict.keys():
        print(nodeID)
        if (nodes_dict[nodeID] == -1):
            if(len(groups_dict) > 0):
                start_nodes = sorted_coordinates_df["start_node_link"]
                end_nodes = sorted_coordinates_df["end_node_link"]

                index = sorted_coordinates_df[sorted_coordinates_df["nodeID"] == nodeID].index.values[0]

                start_nodes = ast.literal_eval(start_nodes[index])
                end_nodes = ast.literal_eval(end_nodes[index])

                connected_nodes = start_nodes + end_nodes

                for connected_nodeID in connected_nodes:
                    connected_group_id = nodes_dict[int(connected_nodeID)]
                    if (connected_group_id >= 0):
                        group = groups_dict[connected_group_id]
                        if (len(group) < max_number_of_nodes_in_group):
                            group.append(nodeID)
                            nodes_dict[nodeID] = connected_group_id
                            break

                if(nodes_dict[nodeID] == -1):
                    create_group(nodeID, group_id, groups_dict, nodes_dict)
                    group_id += 1
            else:
                create_group(nodeID, group_id, groups_dict, nodes_dict)
                group_id += 1

    print(groups_dict)

    return groups_dict


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    input_filename = "1M_8_junctions_1_res_with_1_leak_rand_bd_validation_merged.csv"
    output_filename = "1M_8_junctions_1_res_with_1_leak_rand_bd_validation"

    x = "tensorflow_datasets/8_juncs_1_res/temp.csv"

    # process_dataset_for_binary_classification(input_filename, output_filename)

    # process_dataset_for_regression(input_filename, output_filename)

    extract_subgroups_from_epanet_network(x, 2)