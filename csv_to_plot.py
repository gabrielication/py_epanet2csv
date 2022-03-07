import csv
import matplotlib.pyplot as plt

timestamp_axis = []
node_demand_axis = []
reservoir_demand_axis = {} #we need a dictionary of reservoir demand! we have multiple single values for the same timestamp

with open('output.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    node_demand_sum = 0.0

    for line in csv_reader:
        if len(timestamp_axis) == 0: #if this is the first row that we are reading then we can insert the first timestamp
            timestamp_axis.append(line[0]) 
        elif (line[0] != timestamp_axis[-1]): #if this is the next timestamp that we are reading then we have to append the new one and set the other values
            timestamp_axis.append(line[0]) #insert new timestamp
            node_demand_axis.append(node_demand_sum) #we have to save the sum of junctions otherwise it will be lost
            #print(node_demand_sum)
            node_demand_sum = 0.0 #we can reset the sum

        demand = float(line[2])

        if(line[-1] == "Junction"):
            node_demand_sum += demand
        elif(line[-1] == "Reservoir"):
            reservoir_id = line[1]

            #demand = abs(demand)

            if(reservoir_id in reservoir_demand_axis.keys()):
                reservoir_demand_axis[reservoir_id].append(demand)
            else:
                reservoir_demand_axis[reservoir_id] = [demand]
            

    node_demand_axis.append(node_demand_sum) #if we dont do this we lose the last sum

# plotting the points
plt.plot(timestamp_axis, node_demand_axis, label="Junctions Demand")

for key in reservoir_demand_axis:
    plt.plot(timestamp_axis, reservoir_demand_axis[key], label = key)
 
# naming the x axis
plt.xlabel('Timestamp (hour)')
# naming the y axis
plt.ylabel('Demand (GPM)')
 
# giving a title to my graph
plt.title('Demand Graph')
 
# function to show the plot
plt.show()