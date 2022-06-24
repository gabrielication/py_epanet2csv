import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SF7_125 = pd.read_csv ('SF7_Audio.csv')
accuracy_net1 = pd.read_csv ('exported_month_large_complete_one_reservoirs_small/1MONTH_prediction_accuracies.csv')
accuracy_net2 = pd.read_csv ('exported_month_large_complete_one_reservoirs_small/1MONTH_prediction_accuracies.csv')
accuracy_net3 = pd.read_csv ('exported_month_large_complete_one_reservoirs_small/1MONTH_prediction_accuracies.csv')

# time1 = abs(SF7_125['TRASMISSION TIME'].mean())
# SF = ['7/250kHz', '7/125kHz','8','9','10']
# time = [time2,time1,time3,time4,time5,time6,time7,time8,time9,time10]
# data=[["7/250kHz", time2, time7],
#       ["7/125kHz", time1, time6],
#       ["8", time3, time8],
#       ["9", time4, time9],
#       ["10", time5, time10]
#      ]

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

# columns_values = []
# columns_values.append("Network")
# for jj in range(7):
#     columns_values.append(accuracy.loc[jj, "Classificator"])
#
# print(columns_values)
# print(data)
#
# df=pd.DataFrame(data, columns=columns_values)
# # df.set_index('Network')
# df.set_index('Network', drop=True, append=False, inplace=False, verify_integrity=False)
# print(df)

# data1 = [[30, 25, 50, 20],
# [40, 23, 51, 17],
# [35, 22, 45, 19]]
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

plt.bar(X + 0.00, data[0], width = 0.15)
plt.bar(X + 0.15, data[1], width = 0.15)
plt.bar(X + 0.30, data[2],  width = 0.15)
plt.bar(X + 0.45, data[4],  width = 0.15)
plt.bar(X + 0.6, data[5],  width = 0.15)
plt.bar(X + 0.75, data[6],  width = 0.15)
#
# df.plot(x="Network", y=columns_values, kind="bar")

plt.ylabel('Accuracy')
# plt.xticks(rotation=0)
#
# plt.text(-0.3,time2,round(time2, 2))
# plt.text(0,time7,round(time7, 2))
# plt.text(0.7,time1,round(time1, 2))
# plt.text(1,time6,round(time6, 2))
# plt.text(1.7,time3,round(time3, 2))
# plt.text(2,time8,round(time8, 2))
# plt.text(2.7,time4,round(time4, 2))
# plt.text(3,time9,round(time9, 2))
# plt.text(3.7,time5,round(time5, 2))
# plt.text(4,time10,round(time10, 2))
#
# plt.savefig('Trasmission Time for SF.png',dpi=1200)



# plt.xticks(x_pos, SF)

# plt.ylabel('Trasmission Time [s]')
plt.xlabel('Network')
#
# plt.title('Trasmission Time for SF')
plt.legend()
plt.savefig('accuracy_comparison.png',dpi=1200)

plt.show()

