import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# comparison all model
# accuracy_net1 = pd.read_csv ('ml_results/all_classificators_results/1WEEK_prediction_accuracies.csv')
accuracy_net2_no_sensors = pd.read_csv ('ml_results/all_classificators_results/1MONTH_prediction_accuracies.csv')
# accuracy_net3 = pd.read_csv ('ml_results/all_classificators_results/1YEAR_prediction_accuracies.csv')

# comparison good model
# accuracy_net1 = pd.read_csv ('ml_results/with_sensors_decision_tree_results/1WEEK_prediction_accuracies.csv')
accuracy_net2_sensors = pd.read_csv (
    'ml_results/with_sensors_decision_tree_results/1MONTH_decision_tree_accuracies.csv')
# accuracy_net3 = pd.read_csv ('ml_results/with_sensors_decision_tree_results/1YEAR_prediction_accuracies.csv')


data = []
networks = ["Network 1", "Network 2", "Network 3"]

data_row = []
data_row.append(accuracy_net2_no_sensors.loc[3, "Calc Accuracy"])
data_row.append(0)
data_row.append(0)
data.append(data_row)


data_row = []
data_row.append(accuracy_net2_sensors.loc[0, "1 RES SMALL"])
data_row.append(accuracy_net2_sensors.loc[0, "1 RES LARGE"])
data_row.append(accuracy_net2_sensors.loc[0, "2 RES LARGE"])
data.append(data_row)

print(data)

columns_values = []
# columns_values.append("Network")
for jj in range(7):
    columns_values.append(accuracy_net2_no_sensors.loc[jj, "Classificator"])
#
# print(columns_values)
# print(data)
#
# df=pd.DataFrame(data, columns=columns_values)
# # df.set_index('Network')
# df.set_index('Network', drop=True, append=False, inplace=False, verify_integrity=False)
# print(df)

X = np.arange(3)
print(X)
fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.15)
# ax.bar(X + 0.15, data[1], color = 'g', width = 0.15)
# ax.bar(X + 0.30, data[2], color = 'r', width = 0.15)
# ax.bar(X + 0.45, data[4], color = 'y', width = 0.15)
# ax.bar(X + 0.6, data[5], color = 'k', width = 0.15)
# ax.bar(X + 0.75, data[6], color = 'r', width = 0.15)

plt.bar(X + 0.00, data[0], width = 0.25, label="no sensors")
plt.bar(X + 0.25, data[1], width = 0.25, label="with sensors")

#
# df.plot(x="Network", y=columns_values, kind="bar")

plt.ylabel('Accuracy ['+ columns_values[3] + ']')
plt.yticks(np.arange(0, 1, 0.1))
plt.ylim([0,1])



# plt.savefig('Trasmission Time for SF.png',dpi=1200)
# plt.xticks(x_pos, SF)

# plt.ylabel('Trasmission Time [s]')
# plt.xlabel('Dataset Duration')
plt.xlabel('Network')
#
# plt.title('Trasmission Time for SF')
plt.legend(loc='lower right')
# plt.savefig('accuracy_comparison_all_model.png',dpi=1200)
plt.savefig('accuracy_comparison_good_model.png',dpi=1200)

plt.show()

