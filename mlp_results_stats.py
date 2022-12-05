import pandas as pd

path = "multiple_features_mlp_results_1D.csv"

data = pd.read_csv(path)

# temp = data[data["r2_score"] > 0.98]

print(data["r2_score"].idxmax())
print(data.iloc[19140])
print(data.iloc[19141])

# temp = temp[temp["dataset"] == "1D_ALT_one_res_small_with_leaks_rand_base_dem_nodes_output.csv"]


# id = data["r2_score"].idxmax()
#
# print(data.iloc[id])
# print()
# print(data.iloc[id+1])

