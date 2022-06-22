import csv
import random

import wntr


def pick_rand_smart_sensors(wn):
    node_names = wn.junction_name_list

    len_nodes = int(len(node_names) / 4)
    selected_junctions = random.sample(node_names, len_nodes)

    print(len_nodes)
    print(len(selected_junctions))

    return selected_junctions

def read_junctions(input_file_inp):
    wn = wntr.network.WaterNetworkModel(input_file_inp)

    selected_junctions = pick_rand_smart_sensors(wn)

    return selected_junctions

def assign_smart_sensors(selected_junctions, input_file, prefix_out):

    outName = prefix_out+"nodes_with_sensors.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line in csv_reader:
            node_id = line[1]
            has_leak = line[8]

            smart_sensor_is_present = 0

            if (node_id in selected_junctions):
                if (has_leak == "True"):
                    smart_sensor_is_present = 2
                else:
                    smart_sensor_is_present = 1
            else:
                smart_sensor_is_present = 0

            line.append(smart_sensor_is_present)

            writer.writerow(line)

    out.close()

if __name__ == "__main__":

    input_file_inp = "networks/exported_month_large_complete_one_reservoirs_small.inp"

    selected_junctions= read_junctions(input_file_inp)

    assign_smart_sensors(selected_junctions,
                         "exported_month_large_complete_one_reservoirs_small/1WEEK_nodes_output.csv", "1WEEK_")
    assign_smart_sensors(selected_junctions,
                         "exported_month_large_complete_one_reservoirs_small/1MONTH_nodes_output.csv", "1MONTH_")
    assign_smart_sensors(selected_junctions,
                         "exported_month_large_complete_one_reservoirs_small/1YEAR_nodes_output.csv", "1YEAR_")
