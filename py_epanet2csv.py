import sys
import argparse
import csv

def read_coordinates_from_inp(coord_file):
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
                    node_coordinates[splitted[0]] = [splitted[1],splitted[2]]
                    #print(node_coordinates[splitted[0]])
                    line = network_file.readline()
                break
    return node_coordinates

def to_csv(input_file, coord_file=""):
    found = False
    match = "Node Results"  # We use this string as a filter to find the tables for Node Values
    node_coordinates = {}

    if(len(coord_file)>0):
        node_coordinates = read_coordinates_from_inp(coord_file)

    table_count = 0  # how many tables we found?

    if input_file == "":
        sys.exit("Input file path cannot be empty")

    f = open(input_file, "r")

    outName = "output.csv"
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
        print("Found %d tables" % table_count)
    else:
        print("Cannot find any Node tables inside input file!")


if __name__ == "__main__":
    # Program description
    text = "This script processes the report file from EPANET 2.2, extracts Node Results and converts them to csv. " \
           "Please type the path of the file you want to convert. "

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)

    # Add long and short argument
    parser.add_argument("--input", "-i", help="Input file for processing")
    parser.add_argument("--coordinates", "-c", help="Input file to include node coordinates to CSV")

    # Read arguments from the command line
    args = parser.parse_args()

    # Check for --name
    if args.input:
        input_file = args.input
        if args.coordinates:
            coord_file = args.coordinates
            to_csv(input_file,coord_file)
        else:
            to_csv(input_file)
    else:
        print("Usage 'py_epanet2csv.py -i inputfile'")
