import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# DATASET = "1W_WO"
# DATASET = "1W_W"
# DATASET = "1M_W"

# DATASET = "1W_WO_7_day_fold"
# DATASET = "1W_W_7_day_fold"

# DATASET = "1M_WO_4_week_fold"
DATASET = "1M_W_4_week_fold"

if DATASET == "1W_WO":
    accuracy_net1 = pd.read_csv ('1W_one_res_small/without_sensors/1W_one_res_small_prediction_accuracies.csv')
    accuracy_net2 = pd.read_csv ('1W_one_res_large/without_sensors/1W_one_res_large_prediction_accuracies.csv')
    accuracy_net3 = pd.read_csv ('1W_two_res_large/without_sensors/1W_two_res_large_prediction_accuracies.csv')

if DATASET == "1W_W":
    accuracy_net1 = pd.read_csv ('1W_one_res_small/with_sensors/1W_one_res_small_prediction_accuracies.csv')
    accuracy_net2 = pd.read_csv ('1W_one_res_large/with_sensors/1W_one_res_large_prediction_accuracies.csv')
    accuracy_net3 = pd.read_csv ('1W_two_res_large/with_sensors/1W_two_res_large_prediction_accuracies.csv')

if DATASET == "1M_W":
    accuracy_net1 = pd.read_csv ('1M_one_res_small/with_sensors/1M_one_res_small_prediction_accuracies.csv')
    accuracy_net2 = pd.read_csv ('1M_one_res_large/with_sensors/1M_one_res_large_prediction_accuracies.csv')
    accuracy_net3 = pd.read_csv ('1M_two_res_large/with_sensors/1M_two_res_large_prediction_accuracies.csv')

if DATASET == "1W_WO_7_day_fold":
    accuracy_net1 = pd.read_csv ('1W_one_res_small_7_fold_day/without_sensors/1W_one_res_small_prediction_accuracies_complete.csv')
    accuracy_net2 = pd.read_csv ('1W_one_res_large_7_fold_day/without_sensors/1W_one_res_large_prediction_accuracies_complete.csv')
    accuracy_net3 = pd.read_csv ('1W_two_res_large_7_fold_day/without_sensors/1W_two_res_large_prediction_accuracies_complete.csv')

if DATASET == "1W_W_7_day_fold":
    accuracy_net1 = pd.read_csv ('1W_one_res_small_7_fold_day/with_sensors/1W_one_res_small_prediction_accuracies_complete.csv')
    accuracy_net2 = pd.read_csv ('1W_one_res_large_7_fold_day/with_sensors/1W_one_res_large_prediction_accuracies_complete.csv')
    accuracy_net3 = pd.read_csv ('1W_two_res_large_7_fold_day/with_sensors/1W_two_res_large_prediction_accuracies_complete.csv')

if DATASET == "1M_WO_4_week_fold":
    accuracy_net1 = pd.read_csv ('1M_one_res_small_4_fold_week/without_sensors/1M_one_res_small_prediction_accuracies_complete.csv')
    accuracy_net2 = pd.read_csv ('1M_one_res_large_4_fold_week/without_sensors/1M_one_res_large_prediction_accuracies_complete.csv')
    accuracy_net3 = pd.read_csv ('1M_two_res_large_4_fold_week/without_sensors/1M_two_res_large_prediction_accuracies_complete.csv')

if DATASET == "1M_W_4_week_fold":
    accuracy_net1 = pd.read_csv ('1M_one_res_small_4_fold_week/with_sensors/1M_one_res_small_prediction_accuracies_complete.csv')
    accuracy_net2 = pd.read_csv ('1M_one_res_large_4_fold_week/with_sensors/1M_one_res_large_prediction_accuracies_complete.csv')
    accuracy_net3 = pd.read_csv ('1M_two_res_large_4_fold_week/with_sensors/1M_two_res_large_prediction_accuracies_complete.csv')


data = []
networks = ["A", "B", "C"]
for ii in range(6):
    data_row = []
    # data_row.append(networks[ii])
    data_row.append(accuracy_net1.loc[ii, "Accuracy"]*100)
    data_row.append(accuracy_net2.loc[ii, "Accuracy"]*100)
    data_row.append(accuracy_net3.loc[ii, "Accuracy"]*100)
    data.append(data_row)

print(data)
columns_values = []
# for jj in range(6):
#     columns_values.append(accuracy_net1.loc[jj, "Classificator"])
#
print(columns_values)
columns_values = ['KNeighborsClass.', 'LinearSVM', 'RBFSVM', 'DecisionTree', 'RandomForest', 'AdaBoostClass.']
# print(data)
# df=pd.DataFrame(data, columns=columns_values)
# # df.set_index('Network')
# df.set_index('Network', drop=True, append=False, inplace=False, verify_integrity=False)
# print(df)

X = np.arange(3)
print(X)
fig = plt.figure()

barWidth = 0.1

# Set position of bar on X axis
r1 = [0.4, 1.1, 1.8]
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
# r7 = [x + barWidth for x in r6]

plt.bar(r1, data[0], width = 0.10, label=columns_values[0])
plt.bar(r2, data[1], width = 0.10, label=columns_values[1])
plt.bar(r3, data[2],  width = 0.10, label=columns_values[2])
plt.bar(r4, data[3],  width = 0.10, label=columns_values[3])
plt.bar(r5, data[4],  width = 0.10, label=columns_values[4])
plt.bar(r6, data[5],  width = 0.10, label=columns_values[5])
# plt.bar(r7, data[6],  width = 0.10, label=columns_values[6])
#
# df.plot(x="Network", y=columns_values, kind="bar")

plt.ylabel('Accuracy [%]', fontsize=15)
plt.yticks(np.arange(0, 110, 10))
plt.ylim([0,100])


# Add xticks on the middle of the group bars
plt.xticks([r + (barWidth*2.5) for r in r1], ['A', 'B', 'C'])
# plt.xticks(x_pos, SF)
plt.xlabel('Network', fontsize=15)

#
# plt.title('Trasmission Time for SF')
plt.legend(loc='lower right', fontsize=14, ncol=2)

if DATASET == "1W_WO":
    plt.savefig('accuracy_comparison_all_model_1W_WO.png',dpi=1200)
if DATASET == "1W_WO_7_day_fold":
    plt.savefig('accuracy_comparison_all_model_1W_WO_day_fold.png',dpi=1200)

if DATASET == "1W_W":
    plt.savefig('accuracy_comparison_all_model_1W_W.png',dpi=1200)
if DATASET == "1W_W_7_day_fold":
    plt.savefig('accuracy_comparison_all_model_1W_W_day_fold.png',dpi=1200)

if DATASET == "1M_W":
    plt.savefig('accuracy_comparison_all_model_1M_W.png',dpi=1200)
if DATASET == "1M_WO_4_week_fold":
    plt.savefig('accuracy_comparison_all_model_1M_WO_weak_fold.png',dpi=1200)
if DATASET == "1M_W_4_week_fold":
    plt.savefig('accuracy_comparison_all_model_1M_W_weak_fold.png',dpi=1200)

plt.show()

