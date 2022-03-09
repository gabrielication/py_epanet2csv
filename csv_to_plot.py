import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np

filename_input = ""

def plot_node_and_reservoir_demand(filename):
    timestamp_axis = []
    node_demand_axis = []
    reservoir_demand_axis = {}  # we need a dictionary of reservoir demand! we have multiple single values for the same timestamp
    tank_demand_axis = {}  # we need a dictionary of tank demand! we have multiple single values for the same timestamp

    junctions_total_sum = 0

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        node_demand_sum = 0.0
        total_sum = 0.0

        for line in csv_reader:
            if len(timestamp_axis) == 0:
                # if this is the first row that we are reading then we can insert the first timestamp
                timestamp_axis.append(line[0])
            elif (line[0] != timestamp_axis[-1]):
                # if this is the next timestamp that we are reading then we have to append the new one and set the other values
                # insert new timestamp
                timestamp_axis.append(line[0])
                # we have to save the sum of junctions otherwise it will be lost
                node_demand_axis.append(node_demand_sum)
                # we can reset the sum
                #print(total_sum)
                node_demand_sum = 0.0
                junctions_total_sum = 0

            demand = float(line[2])

            total_sum += demand

            if (line[-1] == "Junction"):
                node_demand_sum += demand
                junctions_total_sum += 1
            elif (line[-1] == "Reservoir"):
                reservoir_id = line[1]

                # demand = abs(demand)

                if (reservoir_id in reservoir_demand_axis.keys()):
                    reservoir_demand_axis[reservoir_id].append(demand)
                else:
                    reservoir_demand_axis[reservoir_id] = [demand]
            elif (line[-1] == "Tank"):
                tank_id = line[1]
                if (tank_id in tank_demand_axis.keys()):
                    tank_demand_axis[tank_id].append(demand)
                else:
                    tank_demand_axis[tank_id] = [demand]

        node_demand_axis.append(node_demand_sum)  # if we dont do this we lose the last sum

    # plotting the points
    plt.plot(timestamp_axis, node_demand_axis, label="Junctions (# "+str(junctions_total_sum)+") Total Demand")

    print("total junctions: "+str(junctions_total_sum))

    for key in reservoir_demand_axis:
        plt.plot(timestamp_axis, reservoir_demand_axis[key], label="Reservoir ID: "+key)

    for key in tank_demand_axis:
        plt.plot(timestamp_axis, tank_demand_axis[key], label="Tank ID: "+key)

    # naming the x axis
    plt.xlabel('Timestamp (hour)')
    # naming the y axis
    plt.ylabel('Demand (GPM)')

    # giving a title to my graph
    plt.title('Demand Graph')

    plt.legend()

    # function to show the plot
    plt.show()

def plot_demand_and_pressure(filename):
    node1 = "11"
    node2 = "23"

    timestamp_axis = []
    node1_demand_axis = []
    node2_demand_axis = []

    node1_pressure_axis = []
    node2_pressure_axis = []

    tank_demand_axis = {}  # we need a dictionary of reservoir demand! we have multiple single values for the same timestamp
    tank_pressure_axis = {}  # we need a dictionary of reservoir demand! we have multiple single values for the same timestamp

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line in csv_reader:
            if len(timestamp_axis) == 0:
                # if this is the first row that we are reading then we can insert the first timestamp
                timestamp_axis.append(line[0])
            elif (line[0] != timestamp_axis[-1]):
                timestamp_axis.append(line[0])

            demand = float(line[2])
            pressure = float(line[4])

            if (line[-1] == "Junction"):
                node_id = line[1]

                if (node_id == node1):
                    node1_demand_axis.append(demand)
                    node1_pressure_axis.append(pressure)
                elif (node_id == node2):
                    node2_demand_axis.append(demand)
                    node2_pressure_axis.append(pressure)

            elif (line[-1] == "Tank"):
                tank_id = line[1]

                # demand = abs(demand)

                if (tank_id in tank_demand_axis.keys()):
                    tank_demand_axis[tank_id].append(demand)
                else:
                    tank_demand_axis[tank_id] = [demand]

                if (tank_id in tank_pressure_axis.keys()):
                    tank_pressure_axis[tank_id].append(pressure)
                else:
                    tank_pressure_axis[tank_id] = [pressure]

    #for key in tank_demand_axis:
    #    plt.plot(timestamp_axis, tank_demand_axis[key], label=key)

    if(len(node1_demand_axis) == 0):
        print("Node ID: "+node1+" not found")
        quit()
    elif(len(node2_demand_axis) == 0):
        print("Node ID: "+node2+" not found")
        quit()

    fig, axs = plt.subplots(2)
    #fig.suptitle('Demand and Pressure subplots')

    axs[0].plot(timestamp_axis, node1_demand_axis, label="Junction ID: "+node1)
    axs[0].plot(timestamp_axis, node2_demand_axis, label="Junctions ID: "+node2)
    for key in tank_demand_axis:
        axs[0].plot(timestamp_axis, tank_demand_axis[key], label="Tank ID: "+key)

    axs[0].set_title("Demand for Junction " + node1 + " and " + node2 + " and Tanks")
    axs[0].legend()

    axs[1].plot(timestamp_axis, node1_pressure_axis, label="Junction ID: "+node1)
    axs[1].plot(timestamp_axis, node2_pressure_axis, label="Junction ID: "+node2)
    for key in tank_demand_axis:
        axs[1].plot(timestamp_axis, tank_pressure_axis[key], label="Tank ID: "+key)

    axs[1].legend()
    axs[1].set_title("Pressure for Junction " + node1 + " and " + node2+" and Tanks")

    plt.legend()

    # function to show the plot
    plt.show()

    print(" ")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Program description
    text = "This script processes the report file from EPANET 2.2, extracts Node Results and converts them to csv. " \
           "Please type the path of the file you want to convert. "

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)

    # Add long and short argument
    parser.add_argument("--input", "-i", help="Input file for processing")
    parser.add_argument("--mode", "-m", help="Mode to choose: 'nd': Plot Junction and Reservoir demands, 'dp': Plot Demand and Pressure")

    # Read arguments from the command line
    args = parser.parse_args()

    # Check for --name
    if args.input and args.mode:
        filename_input = args.input
        if args.mode == "nd":
            plot_node_and_reservoir_demand(filename_input)
        elif args.mode == "dp":
            plot_demand_and_pressure(filename_input)
        else:
            print("Mode not supported. See 'csv_to_plot.py -h'")
    else:
        print("Usage 'csv_to_plot.py -i inputfile -m mode'")