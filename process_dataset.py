import pandas as pd
import warnings
import numpy as np

from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv(
    "1M_one_res_small_fixed_leaks_rand_bd_merged.csv")

columns = df[df["hour"] == "0:00:00"]["nodeID"].array

encoder = OneHotEncoder(sparse=False)

encoded = encoder.fit_transform(np.array(columns).reshape(-1, 1))

max_hour = int(df.iloc[-1]["hour"].split(":")[0])

out_dict = {}

for hour in range(max_hour):
    timestamp = str(hour) + ":00:00"

    print("Processing timestamp: ", timestamp)

    leaks_temp = []

    # if "hour" not in out_dict:
    #     out_dict["hour"] = [timestamp]
    # else:
    #     out_dict["hour"].append(timestamp)

    # row = []

    # row.append(timestamp)

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
out_df.to_csv("conv1d_transposed_dataset.csv",index=False)

out_df.to_pickle('conv1d_transposed_dataset.pickle')