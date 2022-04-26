import sys
import argparse
import csv

def read_coordinates_from_inp(coord_file):
    print("Reading coordinates...")

    node_coordinates = {}
    match = "[COORDINATES]"

    with open(coord_file) as network_file:
        for line in network_file:
            if match in line:
                line = network_file.readline()
                line = network_file.readline()

                while(line):
                    splitted = line.split()

                    if(len(splitted) == 0):
                        break

                    node_id = splitted[0]
                    node_x = splitted[1]
                    node_y = splitted[2]

                    node_coordinates[node_id] = [node_x, node_y]

                    line = network_file.readline()
                break

    if(len(node_coordinates) == 0):
        print("No coordinates found!")

    return node_coordinates

def read_pipes_start_end_nodes_from_inp(coord_file):
    print("Reading pipes...")

    start_end_nodes = {}
    match = "[PIPES]"

    with open(coord_file) as network_file:
        for line in network_file:
            if match in line:
                line = network_file.readline()
                line = network_file.readline()

                while (line):
                    splitted = line.split()

                    if (len(splitted) == 0):
                        break

                    pipe_id = splitted[0]
                    start_node = splitted[1]
                    end_node = splitted[2]

                    start_end_nodes[pipe_id] = [start_node, end_node]

                    line = network_file.readline()
                break

    if (len(start_end_nodes) == 0):
        print("No pipes' start end nodes found!")

    return start_end_nodes

def read_pumps_start_end_nodes_from_inp(coord_file):
    print("Reading pumps...")

    start_end_nodes = {}
    match = "[PUMPS]"

    with open(coord_file) as network_file:
        for line in network_file:
            if match in line:
                line = network_file.readline()
                line = network_file.readline()

                while (line):
                    splitted = line.split()

                    if (len(splitted) == 0):
                        break

                    pump_id = splitted[0]
                    start_node = splitted[1]
                    end_node = splitted[2]

                    start_end_nodes[pump_id] = [start_node, end_node]

                    line = network_file.readline()
                break

    if (len(start_end_nodes) == 0):
        print("No pumps' start end nodes found!")

    return start_end_nodes

def nodes_to_csv(input_file, coord_file=""):
    print("Extracting Nodes, can take a while. Please wait...")

    found = False
    match = "Node Results"  # We use this string as a filter to find the tables for Node Values
    node_coordinates = {}

    if(len(coord_file)>0):
        node_coordinates = read_coordinates_from_inp(coord_file)

    table_count = 0  # how many tables we found?

    if input_file == "":
        sys.exit("Input file path cannot be empty")

    f = open(input_file, "r")

    outName = "nodes_output.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    line = f.readline()

    while line:
        if match in line:
            found = True
            table_count += 1
            splitted = line.split()
            hour = splitted[3]  # store the timestamp

            line = f.readline()  # need to skip table headers (5 lines)
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()

            splitted = line.split()

            while (len(splitted) > 0):
                nodeID = splitted[0]
                demand = splitted[1]
                head = splitted[2]
                pressure = splitted[3]
                # chlorine = splitted[4]
                type = ""
                x_coord = ""
                y_coord = ""

                if (splitted[-1] == "Reservoir"):
                    type = "Reservoir"
                elif(splitted[-1] == "Tank"):
                    type = "Tank"
                else:
                    type = "Junction"

                output_row = [hour, nodeID, demand, head, pressure]

                if(len(node_coordinates)>0):
                    if(nodeID not in node_coordinates):
                        print("Wrong coordinate file")
                        break
                    x_coord = node_coordinates[nodeID][0]
                    y_coord = node_coordinates[nodeID][1]
                    output_row.append(x_coord)
                    output_row.append(y_coord)

                output_row.append(type)

                writer.writerow(output_row)

                line = f.readline()
                splitted = line.split()

        line = f.readline()

    out.close()
    f.close()

    if found:
        print("Found %d Node tables" % table_count)
    else:
        print("Cannot find any Node tables inside input file!")

def links_to_csv(input_file, pipes_file=""):
    print("Extracting Links, can take a while. Please wait...")

    match = "Link Results"

    outName = "links_output.csv"
    pipes_start_end_nodes = {}
    pumps_start_end_nodes = {}

    if (len(pipes_file) > 0):
        pipes_start_end_nodes = read_pipes_start_end_nodes_from_inp(coord_file)
        pumps_start_end_nodes = read_pumps_start_end_nodes_from_inp(coord_file)

    out = open(outName, "w")
    writer = csv.writer(out)

    with open(input_file) as pipes_file:
        for line in pipes_file:
            if match in line:
                hour = line.split()[3]

                line = pipes_file.readline()
                line = pipes_file.readline()
                line = pipes_file.readline()
                line = pipes_file.readline()
                line = pipes_file.readline()

                while (line):
                    splitted = line.split()

                    if (len(splitted) == 0):
                        break

                    pipe_id = splitted[0]
                    pipe_flow = splitted[1]
                    pipe_velocity = splitted[2]
                    pipe_headloss = splitted[3]
                    start_node = ""
                    end_node = ""

                    type = splitted[-1]

                    if(type != "Pump"):
                        type = "Pipe"

                    if(len(pipes_start_end_nodes) > 0 and type == "Pipe"):
                        start_node = pipes_start_end_nodes[pipe_id][0]
                        end_node = pipes_start_end_nodes[pipe_id][1]
                    elif(len(pumps_start_end_nodes) > 0 and type == "Pump"):
                        start_node = pumps_start_end_nodes[pipe_id][0]
                        end_node = pumps_start_end_nodes[pipe_id][1]

                    output_row = [hour,pipe_id,pipe_flow,pipe_velocity,pipe_headloss,start_node,end_node,type]

                    writer.writerow(output_row)

                    line = pipes_file.readline()

    out.close()

if __name__ == "__main__":
    # Program description
    text = "This program processes the report file from EPANET 2.2, extracts Node Results and converts them to csv. " \
           "Please type the path of the file you want to convert. "

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)

    # Add long and short argument
    parser.add_argument("--input", "-i", help="Input file for processing")
    parser.add_argument("--coordinates", "-c", help="Input file to include node coordinates to CSV (it is the original network file)")

    # Read arguments from the command line
    args = parser.parse_args()

    # Check for --name
    if args.input:
        input_file = args.input
        if args.coordinates:
            coord_file = args.coordinates
            nodes_to_csv(input_file, coord_file)
            links_to_csv(input_file, coord_file)
        else:
            nodes_to_csv(input_file)
            links_to_csv(input_file)
        print("CSV conversion finished.")
    else:
        print("See 'py_epanet2csv.py -h' in order to know how to correctly execute this program")
