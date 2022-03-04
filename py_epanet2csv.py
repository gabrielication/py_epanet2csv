import sys
import argparse
import re
import csv

MATCH = "Node Results" #We use this string as a filter to find the tables for Node Values
INPUT_FILE_PATH = ""

def toCSV():
    found = False

    table_count = 0 #how many tables we found?

    if INPUT_FILE_PATH == "":
        sys.exit("Input file path cannot be empty")

    f = open(INPUT_FILE_PATH, "r")

    outName = "output.csv"
    out = open(outName,"w")
    writer = csv.writer(out)

    line = f.readline()

    while line:
        if MATCH in line:
            found = True
            table_count += 1
            splitted = line.split( )
            hour = splitted[3] # store the timestamp

            line = f.readline() # need to skip table headers (5 lines)
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()

            splitted = line.split( )

            while (len(splitted) > 0 and splitted[0].isdigit()):
                nodeID = splitted[0]
                demand = splitted[1]
                head = splitted[2]
                pressure = splitted[3]
                #chlorine = splitted[4]

                splitted.insert(0,hour)

                if(len(splitted) == 5):
                    splitted.append("Junction")

                #print(splitted)

                writer.writerow(splitted)

                line = f.readline()
                splitted = line.split( )

        line = f.readline()

    out.close()

    f.close()

    if found:
        print("Found %d tables" % table_count)
    else:
        print("Cannot find any Node tables inside input file!")

if __name__ == "__main__":
# Program description
    text =  "This script processes the report file from EPANET 2.2, extracts Node Results and converts them to csv. Please type the path of the file you want to convert."

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description = text)

    # Add long and short argument
    parser.add_argument("--input", "-i", help="Input file for processing")

    # Read arguments from the command line
    args = parser.parse_args()

    # Check for --name
    if args.input:
        INPUT_FILE_PATH = args.input
        toCSV()
    else:
        print("Usage 'py_epanet2csv.py -i inputfile'")