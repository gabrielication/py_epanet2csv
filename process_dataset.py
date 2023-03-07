import pandas as pd
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv("tensorflow_datasets/one_res_small/gabriele_marzo_2023/1M_one_res_small_fixed_leaks_rand_bd_filtered_merged.csv")

filtered_data = data.groupby('hour').head(10)
base_demands = filtered_data.groupby("hour")["base_demand"].apply(list)
demand_values = filtered_data.groupby("hour")["demand_value"].apply(list)
head_values = filtered_data.groupby("hour")["head_value"].apply(list)
pressure_values = filtered_data.groupby("hour")["pressure_value"].apply(list)
has_leaks = filtered_data.groupby("hour")["has_leak"].apply(list)

df = pd.DataFrame(columns=['list_of_bd', 'list_of_dv', 'list_of_hd', 'list_of_pr', 'has_leak'])

for i in range(0,100):
    timestamp = str(i)+":00:00"
    row = [base_demands.get(timestamp),demand_values.get(timestamp),head_values.get(timestamp),pressure_values.get(timestamp), True in has_leaks.get(timestamp)]

    # add the row to the DataFrame using .append()
    df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

df.to_csv("processed_dataset.csv", index=False)

