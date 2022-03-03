import sys
import argparse
import re

MATCH = "Node Results" #We use this string as a filter to find the tables for Node Values
INPUT_FILE_PATH = ""

def toCSV():
    found = False

    table_count = 0 #how many tables we found?

    if INPUT_FILE_PATH == "":
        sys.exit("Input file path cannot be empty")

    with open(INPUT_FILE_PATH,'r',encoding = 'utf-8') as inputFile: # 'with' closes the file automatically

        for line in inputFile: # read each line of the file
            if MATCH in line: # found a table
                found = True
                table_count += 1
                splitted = line.split( )

                hour = splitted[3] # store the timestamp

                print("Found!",hour)

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