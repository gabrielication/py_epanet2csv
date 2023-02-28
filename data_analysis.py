import pandas as pd
import numpy as np

folder_input = "tensorflow_datasets/"

### 1M no leak
folder_network = "one_res_small/no_leaks_rand_base_demand/1M/"
input_full_dataset = '1M_one_res_small_no_leaks_rand_bd_merged.csv'
input_stat_full_dataset = folder_network + "1W_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"
complete_path = folder_input + input_full_dataset
complete_path_stat = folder_input + input_stat_full_dataset

### 1M leak
folder_network_leakage = "one_res_small/1_at_2_leaks_rand_base_demand/1M/"
input_full_dataset_leakage = '1M_one_res_small_leaks_rand_bd_a0005_merged.csv' # '1M_one_res_small_leaks_rand_bd_merged.csv'
input_stat_full_dataset_leakage = folder_network_leakage + "1W_one_res_small_no_leaks_rand_base_dem_nodes_simulation_stats.csv"
complete_path_leakage = folder_input + input_full_dataset_leakage
complete_path_stat_leakage = folder_input + input_stat_full_dataset_leakage

# dfData = pd.read_csv(folder_input+folder_network+input_full_dataset, delimiter=',')
dfData = pd.read_csv(folder_input+folder_network_leakage+input_full_dataset_leakage, delimiter=',')

print(list(dfData.columns.values))

"""
['hour', 'nodeID', 'base_demand', 'demand_value', 'head_value', 'pressure_value', 'x_pos', 'y_pos', 'node_type', 'has_leak', 'leak_area_value', 'leak_discharge_value', 'leak_demand_value', 'tot_junctions_demand', 'tot_leaks_demand', 'tot_network_demand']
"""
csvColumns = ['hour', 'nodeID', 'base_demand', 'demand_value', 'head_value', 'pressure_value', 'x_pos', 'y_pos', 'node_type', 'has_leak', 'leak_area_value', 'leak_discharge_value', 'leak_demand_value', 'tot_junctions_demand', 'tot_leaks_demand', 'tot_network_demand']

totNumPackets = 0

# dfDataAlien = dfData.dropna(subset=['dev_addr'])
# dfDataAlien = dfDataAlien[dfDataAlien['dev_eui'].isnull()]
# print('ALIEN')
# print(dfDataAlien.shape)
# totNumPackets += dfDataAlien.shape[0]
# print('TOTAL # of Packet : ' + str(totNumPackets))
#
# dfData = dfData.dropna(subset=['dev_eui'])
# print('UNI')
# print(dfData.shape)
totNumPackets += dfData.shape[0]
print('TOTAL # of Packet : ' + str(totNumPackets))

print(np.unique(dfData.nodeID))

nodeID_ordered = ["8614", "8600", "8610", "9402", "8598", "8608", "8620", "8616", "4922", "J106", "8618", "8604", "8596", "9410", "8612", "8602", "8606", "5656", "8622",
	"8624", "8626", "8628", "8630", "8644", "8634", "8632", "8636", "8646", "8688", "8640", "8642", "8638", "8698", "8692", "8648", "8690", "8718",
	"8702", "8700", "8694", "8738", "8696", "8740", "8720", "8706", "8704", "8686", "8708", "8660", "8656", "8664", "8662", "8654", "8716", "8650",
	"8746", "8732", "8684", "8668", "8730", "8658", "8678", "8652", "8676", "8714", "8710", "8712", "8682", "8666", "8674", "8742", "8680", "8672",
	"8792", "8722", "8726", "8724", "8744", "8736", "8728", "8670", "8734", "7384"]

ii = 0
df2 = pd.DataFrame(columns=csvColumns)

# for d in np.unique(dfData.nodeID):
for d in nodeID_ordered:
	print('UNI - ' + str(ii) + ' : ' + str(d))
	# if str(d)=='7384':
	# 	print('stop')

	if not d == '':
		if dfData.loc[dfData.nodeID == str(d)].shape[0] > 1:
			# labels.append(d[-4:])
			# extract relevant info
			df = dfData.loc[dfData.nodeID == str(d)]
			df.reset_index(inplace=True, drop=True)


			# # for index, row in df.iterrows():
			#
			df.demand_value.mean()

			df3 = pd.DataFrame([[df.loc[0,'hour'], df.loc[0,'nodeID'],
								 df.base_demand.mean(), df.demand_value.mean(), df.loc[0,'head_value'],
								 df.pressure_value.mean(), df.loc[0,'x_pos'], df.loc[0,'y_pos'], df.loc[0,'node_type'],
								 df.loc[0,'has_leak'], df.loc[0,'leak_area_value'], df.loc[0,'leak_discharge_value'],
								 df.loc[0,'leak_demand_value'], df.loc[0,'tot_junctions_demand'], df.loc[0,'tot_leaks_demand'],
								 df.loc[0,'tot_network_demand']]
								], columns=csvColumns)

			df2 = df2.append(df3)
			# # df2 = pd.concat([df2, df3], ignore_index=True, sort=False)
			#
			#
			# # if csvIndex > 20:
			# # 	break
			#

			ii += 1

df2.reset_index(inplace=True, drop=True)
# df2 = df2[df2['value_minutes'] > 10 ]
# df2.reset_index(inplace=True, drop=True)

print(df2.shape)
# df2.to_csv(folder_input + folder_network + 'mean_' + input_full_dataset, float_format='%.8f', index=False)
df2.to_csv(folder_input+folder_network_leakage+'mean_'+input_full_dataset_leakage, float_format='%.8f', index=False)
# if csvIndex > 20:
	# 	break
