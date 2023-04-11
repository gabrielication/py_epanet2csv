import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("tensorflow_datasets/one_res_small/gabriele_marzo_2023/1M_one_res_small_rand_leaks_rand_fixed_bd_with_multipliers_merged.csv")

columns = df[df["hour"] == "0:00:00"]["nodeID"].array

max_hour = int(df.iloc[-1]["hour"].split(":")[0])

out_dict = {}

for hour in range(4):
    timestamp = str(hour) + ":00:00"

    print("Processing timestamp: ", timestamp)

    leaks_temp = []
    for nodeID in columns:
        # print(timestamp, nodeID)

        temp = df[df["hour"] == timestamp]
        temp = temp[temp["nodeID"] == nodeID]

        base_demand = float(temp.iloc[-1]["base_demand"])
        demand_value = float(temp.iloc[-1]["demand_value"])
        head_value = float(temp.iloc[-1]["head_value"])
        pressure_value = float(temp.iloc[-1]["pressure_value"])
        x_pos = float(temp.iloc[-1]["x_pos"])
        y_pos = float(temp.iloc[-1]["y_pos"])

        has_leak = int(temp.iloc[-1]["has_leak"])
        leaks_temp.append(has_leak)

        output = [base_demand, demand_value, head_value, pressure_value, x_pos, y_pos]

        if nodeID not in out_dict:
            out_dict[nodeID] = [output]
        else:
            out_dict[nodeID].append(output)

    if "has_leak" not in out_dict:
        out_dict["has_leak"] = [leaks_temp]
    else:
        out_dict["has_leak"].append(leaks_temp)

out_df = pd.DataFrame(out_dict)
# out_df.to_csv("transposed_dataset.csv",index=False)

out_df.to_pickle('filknkne.pickle')