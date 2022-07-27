import csv

def how_many_gws(input_file):
    result = 0

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line in csv_reader:
            result = (len(line) - 13) / 2 #there are 13 columns before the first gw couple
            break

    result = int(result)

    print(input_file+" contains "+str(result)+" gateways\n")

    return result

def create_csv_per_gw(input_file):

    number_of_gw = how_many_gws(input_file)

    for gw_id in range(number_of_gw):

        outName = "gw_"+str(gw_id)+"_"+input_file

        out = open(outName, "w", newline='', encoding='utf-8')

        writer = csv.writer(out)

        write_gw_csv(input_file, writer, gw_id)

        out.close()

    print("Finished\n\n")

def write_gw_csv(input_file, writer, gw_id):

    print("Writing csv for GW id: "+str(gw_id)+"...")

    gw_rssi_index = 13 + (gw_id * 2)
    gw_sf_index = gw_rssi_index + 1

    header = ["hour", "nodeID", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
							  "node_type", "has_leak", "leak_area_value", "leak_discharge_value",
							  "current_leak_demand_value", "gw_rssi", "gw_sf"]

    writer.writerow(header)

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line in csv_reader:

            gw_rssi = line[gw_rssi_index]
            gw_sf = line[gw_sf_index]

            output_row = line[:12] #we don't care about sensor flag

            if(float(gw_sf) != 7.0):
                output_row[2] = "0.0"
                output_row[3] = "0.0"
                output_row[4] = "0.0"
                output_row[5] = "0.0"
                output_row[6] = "0.0"

                #output_row[8] = "False"

                output_row[9] = "0.0"
                output_row[10] = "0.0"
                output_row[11] = "0.0"

            output_row.append(gw_rssi)
            output_row.append(gw_sf)

            writer.writerow(output_row)

    print("csv written!\n")

if __name__ == "__main__":

    create_csv_per_gw("lora1D_one_res_small_nodes_output.csv")
    create_csv_per_gw("lora1M_one_res_small_nodes_output.csv")
    create_csv_per_gw("lora1M_one_res_large_nodes_output.csv")
    create_csv_per_gw("lora1M_two_res_large_nodes_output.csv")