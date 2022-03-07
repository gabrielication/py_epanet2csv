import sys
import argparse
import re
import csv


# importing the required module
import matplotlib.pyplot as plt

timestamp_axis = []
node_demand_axis = []
reservoir_demand_axis = []

with open('output.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    node_demand_sum = 0.0

    for line in csv_reader:
        if len(timestamp_axis) == 0:
            timestamp_axis.append(line[0])
        elif (line[0] != timestamp_axis[-1]):
            timestamp_axis.append(line[0])
            node_demand_axis.append(node_demand_sum)
            #print(node_demand_sum)
            node_demand_sum = 0.0

        if(line[-1] == "Junction"):
            node_demand_sum += float(line[2])
    node_demand_axis.append(node_demand_sum)

# plotting the points
plt.plot(timestamp_axis, node_demand_axis, label="Junctions Demand")
 
# naming the x axis
plt.xlabel('Timestamp (hour)')
# naming the y axis
plt.ylabel('Demand (GPM)')
 
# giving a title to my graph
plt.title('Demand Graph')
 
# function to show the plot
plt.show()