import csv

def extract_nodes(input_file, up_x_coord, up_y_coord, down_x_coord, down_y_coord):
    outName = "extracted_nodes.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    #494155.350
    #1379384.780

    #1376590.440

    selected_ids = ["8596"]

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line in csv_reader:
            node_id = line[1]
            node_x = float(line[5])
            node_y = float(line[6])

            if(node_id in selected_ids):
                writer.writerow(line)

            elif(node_x >= up_x_coord):
                if(node_x <= down_x_coord):
                    if(node_y <= up_y_coord):
                        if(node_y >= down_y_coord):
                            writer.writerow(line)

    out.close()

    print("Extracted nodes to "+outName)

if __name__ == "__main__":
    extract_nodes("output.csv",494318.100,1379694.190,500947.500,1376590.440)