import pandas as pd
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("tensorflow_datasets/one_res_small/gabriele_marzo_2023/1M_one_res_small_rand_leaks_rand_fixed_bd_with_multipliers_merged.csv")

columns = df[df["hour"] == "0:00:00"]["nodeID"].array

out_df = pd.DataFrame(columns=[columns])

max_hour = int(df.iloc[-1]["hour"].split(":")[0])

for hour in range(max_hour):
    timestamp = str(hour) + ":00:00"

    row = {}
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

        has_leak = float(temp.iloc[-1]["has_leak"])

        row[nodeID] = [base_demand, demand_value, head_value, pressure_value, x_pos, y_pos]

    print(row)

#
# filtered_data = data.groupby('hour').head(10)
# base_demands = filtered_data.groupby("hour")["base_demand"].apply(list)
# demand_values = filtered_data.groupby("hour")["demand_value"].apply(list)
# head_values = filtered_data.groupby("hour")["head_value"].apply(list)
# pressure_values = filtered_data.groupby("hour")["pressure_value"].apply(list)
# has_leaks = filtered_data.groupby("hour")["has_leak"].apply(list)
#
# df = pd.DataFrame(columns=['list_of_bd', 'list_of_dv', 'list_of_hd', 'list_of_pr', 'has_leak'])
#
# for i in range(0,100):
#     timestamp = str(i)+":00:00"
#     row = [base_demands.get(timestamp),demand_values.get(timestamp),head_values.get(timestamp),pressure_values.get(timestamp), True in has_leaks.get(timestamp)]
#
#     # add the row to the DataFrame using .append()
#     df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
#
# df.to_csv("processed_dataset.csv", index=False)
#
# for i in range(0,100):
#     timestamp = str(i) + ":00:00"