import numpy as np
import pandas as pd
import csv
import random
import warnings

# filter out the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

filename1 = "tensorflow_datasets/one_res_small/gabriele_marzo_2023/1M_one_res_small_no_leaks_rand_bd_merged.csv"
filename2 = "tensorflow_datasets/one_res_small/gabriele_marzo_2023/1M_one_res_small_rand_leaks_rand_bd_merged.csv"

df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)

sim_duration = int(df1['hour'].iloc[-1].split(":")[0])
sim_duration_doubled = sim_duration * 2

last_row_from_first_df = df1.iloc[-1]["hour"]
number_of_nodes = df1['hour'].value_counts()[last_row_from_first_df]

merged_df = pd.DataFrame(columns=df1.columns)

hour_df1 = 0
hour_df2 = 0
hour = 0

while not df1.empty or not df2.empty:
    timestamp = str(hour)+":00:00"
    if random.random() < 0.5:
        if(not df1.empty):
            selected_rows = df1.loc[df1['hour'] == str(hour_df1)+":00:00"]
            merged_df = pd.concat([merged_df, selected_rows], ignore_index=True)

            # get the indices of the rows to drop
            rows_to_drop = df1.index[df1['hour'] == str(hour_df1)+":00:00"].tolist()

            # drop the rows from the dataframe
            df1 = df1.drop(rows_to_drop)

            hour_df1 += 1
        elif(not df2.empty):
            selected_rows = df2.loc[df2['hour'] == str(hour_df2) + ":00:00"]
            merged_df = pd.concat([merged_df, selected_rows], ignore_index=True)

            # get the indices of the rows to drop
            rows_to_drop = df2.index[df2['hour'] == str(hour_df2) + ":00:00"].tolist()

            # drop the rows from the dataframe
            df2 = df2.drop(rows_to_drop)

            hour_df2 += 1
    else:
        if (not df2.empty):
            selected_rows = df2.loc[df2['hour'] == str(hour_df2) + ":00:00"]
            merged_df = pd.concat([merged_df, selected_rows], ignore_index=True)

            # get the indices of the rows to drop
            rows_to_drop = df2.index[df2['hour'] == str(hour_df2) + ":00:00"].tolist()

            # drop the rows from the dataframe
            df2 = df2.drop(rows_to_drop)

            hour_df2 += 1
        elif (not df1.empty):
            selected_rows = df1.loc[df1['hour'] == str(hour_df1) + ":00:00"]
            merged_df = pd.concat([merged_df, selected_rows], ignore_index=True)

            # get the indices of the rows to drop
            rows_to_drop = df1.index[df1['hour'] == str(hour_df1) + ":00:00"].tolist()

            # drop the rows from the dataframe
            df1 = df1.drop(rows_to_drop)

            hour_df1 += 1

hour = 0

for mult in range(1, sim_duration_doubled+1):
    stop = mult * number_of_nodes
    start = stop - number_of_nodes
    value = str(hour)+":00:00"
    merged_df.loc[start:stop, 'hour'] = value

    hour+=1

merged_df.to_csv("2M_merged.csv",index=False)



