import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#test

# comparison all model
accuracy_net1 = pd.read_csv ('ml_results/all_classificators_results/1WEEK_prediction_accuracies.csv')
accuracy_net2 = pd.read_csv ('ml_results/all_classificators_results/1MONTH_prediction_accuracies.csv')
accuracy_net3 = pd.read_csv ('ml_results/all_classificators_results/1YEAR_prediction_accuracies.csv')

data = []
networks = ["Network 1", "Network 2", "Network 3"]
for ii in range(7):
    data_row = []
    # data_row.append(networks[ii])
    data_row.append(accuracy_net1.loc[ii, "Calc Accuracy"])
    data_row.append(accuracy_net2.loc[ii, "Calc Accuracy"])
    data_row.append(accuracy_net3.loc[ii, "Calc Accuracy"])
    data.append(data_row)

print(data)
columns_values = []
# columns_values.append("Network")
for jj in range(7):
    columns_values.append(accuracy_net1.loc[jj, "Classificator"])
#
# print(columns_values)
# print(data)
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

plt.bar(X + 0.00, data[0], width = 0.10, label=columns_values[0])
plt.bar(X + 0.1, data[1], width = 0.10, label=columns_values[1])
plt.bar(X + 0.2, data[2],  width = 0.10, label=columns_values[2])
plt.bar(X + 0.3, data[3],  width = 0.10, label=columns_values[3])
plt.bar(X + 0.4, data[4],  width = 0.10, label=columns_values[4])
plt.bar(X + 0.5, data[5],  width = 0.10, label=columns_values[5])
plt.bar(X + 0.6, data[6],  width = 0.10, label=columns_values[6])
#
# df.plot(x="Network", y=columns_values, kind="bar")

plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, 0.1))
plt.ylim([0,1])



# plt.savefig('Trasmission Time for SF.png',dpi=1200)
# plt.xticks(x_pos, SF)

# plt.ylabel('Trasmission Time [s]')
# plt.xlabel('Dataset Duration')
plt.xlabel('Dataset Duration')
#
# plt.title('Trasmission Time for SF')
plt.legend(loc='lower right')
plt.savefig('accuracy_comparison_all_model.png',dpi=1200)

plt.show()

