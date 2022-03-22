import csv

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
    extract_nodes("output.csv",494318.100,1379694.190,500947.500,1376590.440,["8596"])