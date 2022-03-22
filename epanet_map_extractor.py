import csv
import sys
import argparse

def extract_nodes(input_file, min_x_coord, max_y_coord, max_x_coord, min_y_coord, selected_ids=[]):
    outName = "extracted_nodes.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line in csv_reader:
            node_id = line[1]
            node_x = float(line[5])
            node_y = float(line[6])

            if(len(selected_ids) > 0 and node_id in selected_ids):
                writer.writerow(line)

            elif(node_x >= min_x_coord):
                if(node_x <= max_x_coord):
                    if(node_y <= max_y_coord):
                        if(node_y >= min_y_coord):
                            writer.writerow(line)

    out.close()

    print("Extracted nodes to "+outName)

if __name__ == "__main__":
    # Program description
    text = "This program extracts a subset of the network from the CSV file produced from py_epanet2csv.py. Be sure to produce a CSV file with coordinates in it. Works for Junctions ONLY!"

    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text)

    # Add long and short argument
    parser.add_argument("--input", "-i", help="Input junction CSV file from py_epanet2csv.py")
    parser.add_argument("--Xmin", "-xmin", help="Minimum X coordinate")
    parser.add_argument("--Xmax", "-xmax", help="Maximum X coordinate")
    parser.add_argument("--Ymin", "-ymin", help="Minimum Y coordinate")
    parser.add_argument("--Ymax", "-ymax", help="Maximum Y coordinate")

    parser.add_argument("--selectedNode", "-n", help="Include a node to the selection")

    # Read arguments from the command line
    args = parser.parse_args()

    if args.input and args.Xmin and args.Xmax and args.Ymin and args.Ymax:
        input_file = args.input
        x_min = float(args.Xmin)
        y_max = float(args.Ymax)
        x_max = float(args.Xmax)
        y_min = float(args.Ymin)

        if(args.selectedNode):
            node = [args.selectedNode]
            extract_nodes(input_file,x_min,y_max,x_max,y_min,node)
        else:
            extract_nodes(input_file, x_min, y_max, x_max, y_min, [])
    else:
        print("Use 'epanet_map_extractor.py -h'")

    #extract_nodes("junctions_output.csv",494318.100,1379694.190,500947.500,1376590.440,["8596"])

    #python3 epanet_map_extractor.py -i junctions_output.csv -xmin 494318.100 -ymax 1379694.190 -xmax 500947.500 -ymin 1376590.440 -n 8596