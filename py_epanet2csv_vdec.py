import multiprocessing

import wntr
import csv
import sys
from decimal import Decimal
import random

from multiprocessing import Process, Lock

smart_sensors_enabled = False
leaks_enabled = False

lock = Lock()

#MULTIPROCESS GENERATION CURRENTLY DEPRECATED
class WNTR_Process (Process):

    def __init__(self, name, wn, smart_sensor_junctions):
        Process.__init__(self)
        self.name = name
        self.wn = wn
        self.smart_sensor_junctions = smart_sensor_junctions

    def run(self):
        print ("Proc '" + self.name + "' started")
        # Lock on
        lock.acquire()
        # Free lock
        lock.release()
        run_sim(self.wn, self.smart_sensor_junctions, output_file_name=self.name+"_", proc_name=self.name + ": ")

def run_sim(wn, sim_duration, smart_sensor_junctions, number_of_nodes_with_leaks=0, number_of_nodes_with_sensors=0, output_file_name="", proc_name=""):
    print(proc_name + "Configuring simulation...")

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    wn.options.time.duration = sim_duration

    new_pat = create_custom_pattern(wn, "custom_1", 1, 1, 1, sim_duration)
    assign_rand_demand_to_junctions(wn, 0, 5, "custom_1")

    node_names = wn.node_name_list
    link_names = wn.link_name_list

    print(proc_name + "Running simulation...")

    results = [wntr.sim.WNTRSimulator(wn).run_sim()]

    print(proc_name + "Simulation finished. Writing to csv (can take a while)...")

    nodes_to_csv(wn, results, node_names, output_file_name, proc_name, smart_sensor_junctions,number_of_nodes_with_leaks,number_of_nodes_with_sensors)
    #links_to_csv(wn, results, link_names, output_file_name, proc_name)

    print(proc_name + "Finished!")

def run_sim_with_random_base_demands(wn, sim_duration, smart_sensor_junctions, number_of_nodes_with_leaks=0, number_of_nodes_with_sensors=0, output_file_name="", proc_name="", list_of_demands=None):
    print(proc_name + "Configuring simulation...")

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    new_pat = create_custom_pattern(wn, "custom_1", 1, 1, 1, sim_duration)

    sim_duration_in_hours = int(sim_duration / 3600)

    node_names = wn.node_name_list
    link_names = wn.link_name_list

    print(proc_name + "Running simulation...")

    head_temp = create_header_nodes_to_csv(proc_name, output_file_name)

    writer = head_temp[0]
    out = head_temp[1]

    # results = []

    tot_leak_demand = Decimal('0')
    tot_nodes_demand = Decimal('0')

    for hour in range(sim_duration_in_hours):
        wn.options.time.duration = (hour) * 3600

        assign_rand_demand_to_junctions(wn, 0, 5, "custom_1", list_of_demands)

        node_obj = wn.get_node("4922")
        #print(node_obj.demand_timeseries_list[0].base_value)

        results = wntr.sim.WNTRSimulator(wn).run_sim()

        csv_res = write_body_nodes_to_csv_func(wn, node_names, [results], writer, False)

        tot_nodes_demand += csv_res[0]
        tot_leak_demand += csv_res[1]

        #print("hour is ",hour)

    print(proc_name + "Simulation finished. Writing to csv (can take a while)...")

    write_bottom_nodes_to_csv_func(wn, out, tot_nodes_demand, tot_leak_demand, output_file_name, proc_name,
                                   number_of_nodes_with_leaks, number_of_nodes_with_sensors)

    # cheap fix for generating csv is to use its function directly into the simulation... not good but works

    # nodes_to_csv(wn, results, node_names, output_file_name, proc_name, smart_sensor_junctions,number_of_nodes_with_leaks,number_of_nodes_with_sensors)
    # links_to_csv(wn, results, link_names, output_file_name, proc_name)

    print(proc_name + "Finished!")

def pick_rand_leaks(wn, number_of_junctions_with_leaks):
    node_names = wn.junction_name_list

    selected_junctions = random.sample(node_names, number_of_junctions_with_leaks)

    return selected_junctions

def assign_leaks(wn, area_size, selected_junctions, proc_name):
    for node_id in selected_junctions:
        node_obj = wn.get_node(node_id)

        node_obj.add_leak(wn, area=area_size, start_time=0)

        #print(proc_name + "Leak added to node id: ", node_id)

def pick_rand_smart_sensors(wn,number_of_nodes_with_sensors):
    node_names = wn.junction_name_list

    selected_junctions = random.sample(node_names, number_of_nodes_with_sensors)

    return selected_junctions

def create_custom_pattern(wn, name, min, max, step, duration):
    timeops = wntr.network.options.TimeOptions(duration)

    multipliers = []

    for multiplier_step in range(min,max+step,step):
        multipliers.append(multiplier_step)

    out_pattern = wntr.network.Pattern("custom", multipliers, time_options=timeops)

    wn.add_pattern(name, out_pattern)

    return out_pattern

def generate_rand_demand_array(min, max, number_of_junctions, hours_of_simulation):

    output = []

    hours_index = int(hours_of_simulation)

    #print(hours_index)

    for hour in range(0,hours_index):

        demands_in_a_hour = []

        for junc in range(0,number_of_junctions):
            new_demand = random.uniform(min, max)

            demands_in_a_hour.append(new_demand)

        output.append(demands_in_a_hour)

    return output

def assign_rand_demand_to_junctions(wn, min, max, pattern=None, list_of_demands=None):
    node_names = wn.junction_name_list

    i = 0

    for juncID in node_names:
        junc_obj = wn.get_node(juncID)

        if(list_of_demands is None):
            # old = junc_obj.demand_timeseries_list[0].base_value
            new_demand = random.uniform(min, max)

            #print(wn.sim_time / 3600)
        else:
            timestamp = int(wn.sim_time / 3600)

            #print(i,timestamp)

            new_demand = list_of_demands[timestamp][i]

            i += 1

        # junc_obj.demand_timeseries_list[0].base_value = new_demand

        # you can't apparently just change the old pattern like the base demand, it seems to be a read only field.
        # for now we just create a new timeseries and delete the old one
        # junc_obj.demand_timeseries_list[0].pattern = pattern

        junc_obj.add_demand(base=new_demand, pattern_name=pattern)
        del junc_obj.demand_timeseries_list[0]

def links_to_csv(wn, results, link_names, output_file_name, proc_name):
    print(proc_name + "Writing Links' CSV...")

    flow_results = results.link['flowrate']
    velocity_results = results.link['velocity']
    #headloss_results = results.link['headloss'] NOT COMPATIBLE WITH WNTR

    indexes = flow_results.index

    outName = output_file_name+"links_output.csv"
    out = open(outName, "w", newline='', encoding='utf-8')
    writer = csv.writer(out)

    debug = False

    for timestamp in indexes:
        tot_demand = Decimal('0')
        demand_value = Decimal('0')

        #hour = pd.to_datetime(timestamp, unit='s').time()

        hour = str(
            int(timestamp / 3600)) + ":00:00"  # cheap way to calculate timestamps. if we choose an interval != 1h then it will break

        #print(timestamp)
        for linkID in link_names:
            link_obj = wn.get_link(linkID)

            flow_value = Decimal(str(flow_results.loc[timestamp, linkID]))
            flow_value = round(flow_value, 8)

            velocity_value = Decimal(str(velocity_results.loc[timestamp, linkID]))
            velocity_value = round(velocity_value, 8)

            #headloss_value = Decimal(str(headloss_results.loc[timestamp, linkID]))
            #headloss_value = round(headloss_value, 8)

            start_node_value = link_obj.start_node_name
            end_node_value = link_obj.end_node_name

            node_type = link_obj.__class__.__name__

            #output_row = [hour, linkID, flow_value, velocity_value, headloss_value, start_node_value, end_node_value, node_type]
            output_row = [hour, linkID, flow_value, velocity_value, start_node_value, end_node_value,
                          node_type]

            writer.writerow(output_row)

    out.close()

    print(proc_name + "Links' CSV written.")

def create_header_nodes_to_csv(proc_name, output_file_name):
    print(proc_name + "Writing Nodes' CSV...")

    outName = output_file_name + "nodes_output.csv"
    out = open(outName, "w", newline='', encoding='utf-8')
    writer = csv.writer(out)

    header = ["hour", "nodeID", "base_demand", "demand_value", "head_value", "pressure_value", "x_pos", "y_pos",
              "node_type", "has_leak", "leak_area_value", "leak_discharge_value", "current_leak_demand_value",
              "smart_sensor_is_present", "tot_network_demand"]

    writer.writerow(header)

    return writer, out

max_count_is = 0

def write_body_nodes_to_csv_func(wn, node_names, list_results, writer, debug):

    global max_count_is

    for results in list_results:
        demand_results = results.node['demand']
        head_results = results.node['head']
        pressure_results = results.node['pressure']

        indexes = demand_results.index



        for timestamp in indexes:
            tot_demand = Decimal('0')

            tot_leak_demand = Decimal('0')
            tot_nodes_demand = Decimal('0')

            demand_value = Decimal('0')

            # hour = pd.to_datetime(timestamp, unit='s').time()

            hour = str(
                int(timestamp / 3600)) + ":00:00"  # cheap way to calculate timestamps. if we choose an interval != 1h then it will break

            # print(hour)

            count = 0

            for nodeID in node_names:
                node_obj = wn.get_node(nodeID)

                demand_value = Decimal(str(demand_results.loc[timestamp, nodeID]))
                demand_value = round(demand_value, 8)

                if demand_value > 0:
                    count += 1

                max_count_is = max(count, max_count_is)

                head_value = Decimal(str(head_results.loc[timestamp, nodeID]))
                head_value = round(head_value, 8)
                pressure_value = Decimal(str(pressure_results.loc[timestamp, nodeID]))
                pressure_value = round(pressure_value, 8)

                base_demand = 0.0

                node_type = node_obj.__class__.__name__

                if node_type == "Junction":
                    base_demand = Decimal(str(node_obj.demand_timeseries_list[0].base_value))
                    base_demand = round(base_demand, 8)

                    #if(nodeID=="4922"):
                        #print(node_obj.demand_timeseries_list[0].base_value)

                smart_sensor_is_present = 0  # can be 0,1,2

                # has_leak = node_obj._leak #this leak-flag represents if a leak was set to the node but not if the leak is flowing now on this timestamp
                has_leak = False

                leak_area_value = node_obj.leak_area  # I think that this does not require an approximation... right?

                leak_discharge_value = node_obj.leak_discharge_coeff  # Same as above...?

                current_leak_demand_value = results.node["leak_demand"].at[timestamp, nodeID]
                '''
                if(current_leak_demand_value > 0.0):
                    has_leak = True #this leak-flag is set to true if the leak is flowing now on this timestamp
                '''
                if (leak_area_value > 0.0):
                    has_leak = True  # this leak-flag is set to true if we see a hole in the node

                if (smart_sensors_enabled):
                    if (nodeID in smart_sensor_junctions):
                        if (has_leak):
                            smart_sensor_is_present = 2
                        else:
                            smart_sensor_is_present = 1
                    else:
                        smart_sensor_is_present = 0

                current_leak_demand_value = Decimal(str(current_leak_demand_value))
                current_leak_demand_value = round(current_leak_demand_value, 8)

                if debug:
                    print("--------")
                    print(nodeID)
                    print("demand")
                    print("{:4.15}".format(demand_value))
                    print("tot_d")
                    print("{:4.15f}".format(tot_demand))

                tot_demand = tot_demand + demand_value + current_leak_demand_value
                tot_demand = round(tot_demand, 8)

                if debug:
                    print("tot_d")
                    print("{:4.15f}".format(tot_demand))

                if debug:
                    if nodeID == "101":
                        print("test")
                        sys.exit(1)

                x_pos = node_obj.coordinates[0]
                y_pos = node_obj.coordinates[1]

                if (node_type == "Junction"):
                    tot_leak_demand = tot_leak_demand + current_leak_demand_value

                    tot_nodes_demand = tot_nodes_demand + demand_value + current_leak_demand_value

                    if debug:
                        print("Node ID: " + str(nodeID))
                        print("Demand is: " + str(demand_value) + " Leak demand is: " + str(current_leak_demand_value))
                        print("Tot node demand: " + str(tot_nodes_demand))
                        print("Tot leak demand: " + str(tot_leak_demand))
                        print(" ")

                output_row = [hour, nodeID, base_demand, demand_value, head_value, pressure_value, x_pos, y_pos,
                              node_type, has_leak, leak_area_value, leak_discharge_value, current_leak_demand_value,
                              smart_sensor_is_present, tot_demand]

                # print(nodeID)
                # print(demand_value)
                # print(tot_demand)
                # print("Node (" + str(nodeID) + ") demand is: "+str(demand_value)+" so Tot demand is now: " + str(tot_demand))
                # print("Node ({}) demand is: {} so Tot demand is now: {}".format(nodeID,demand_value,tot_demand))

                writer.writerow(output_row)

            # print(timestamp)

            print("count is: ",count," max is: ",max_count_is, " n. junctions: ",wn.num_junctions)

            if debug:
                if tot_demand < 1e-6:
                    print("Tot demand at " + str(hour) + " is: 0 (" + str(tot_demand) + ")")
                else:
                    print("Tot demand at " + str(hour) + " is: " + str(tot_demand))
            # break

    return tot_nodes_demand, tot_leak_demand

def write_bottom_nodes_to_csv_func(wn, out, tot_nodes_demand, tot_leak_demand, output_file_name, proc_name, number_of_nodes_with_leaks, number_of_nodes_with_sensors):

    out.close()

    if (tot_nodes_demand > 0):
        leak_percentage = (tot_leak_demand / tot_nodes_demand) * 100
        leak_percentage = round(leak_percentage, 4)
    else:
        leak_percentage = 0.0

    number_of_nodes = len(wn.node_name_list)
    number_of_junctions = len(wn.junction_name_list)
    number_of_reservoirs = len(wn.reservoir_name_list)
    number_of_tanks = len(wn.tank_name_list)
    time_spent_on_sim = (wn.options.time.duration / 3600) + 1

    outName = output_file_name + "nodes_simulation_stats.csv"
    out = open(outName, "w", newline='', encoding='utf-8')
    writer = csv.writer(out)

    header = ["tot_nodes_demand", "leak_percentage", "number_of_nodes", "number_of_junctions",
              "number_of_reservoirs", "number_of_tanks", "number_of_nodes_with_leaks", "number_of_nodes_with_sensors",
              "time_spent_on_sim"]

    writer.writerow(header)

    print("\n\nTot demand for Nodes only is: " + str(tot_nodes_demand) + " and tot_leak_demand is: " + str(
        tot_leak_demand))
    print("Total leak demand for nodes is:  " + str(leak_percentage) + "% of the Total Nodes' demand")
    print("Number of nodes inside of the network is: " + str(number_of_nodes))
    print("Number of Junctions only: " + str(number_of_junctions))
    print("Number of Reservoirs only: " + str(number_of_reservoirs))
    print("Number of Tanks only: " + str(number_of_tanks))
    print("Number of Junctions with leaks: " + str(number_of_nodes_with_leaks))
    print("Number of Junctions with sensors: " + str(number_of_nodes_with_sensors))
    print("Total hours simulated (remember that 0:00 is included): " + str(time_spent_on_sim) + "\n\n")

    output_row = [tot_nodes_demand, leak_percentage, number_of_nodes, number_of_junctions,
                  number_of_reservoirs, number_of_tanks, number_of_nodes_with_leaks, number_of_nodes_with_sensors,
                  time_spent_on_sim]

    # print(nodeID)
    # print(demand_value)
    # print(tot_demand)
    # print("Node (" + str(nodeID) + ") demand is: "+str(demand_value)+" so Tot demand is now: " + str(tot_demand))
    # print("Node ({}) demand is: {} so Tot demand is now: {}".format(nodeID,demand_value,tot_demand))

    writer.writerow(output_row)

    out.close()

    print(proc_name + "Nodes' CSV written.")

def nodes_to_csv(wn, list_results, node_names, output_file_name, proc_name, smart_sensor_junctions, number_of_nodes_with_leaks, number_of_nodes_with_sensors):
    print(proc_name + "Writing Nodes' CSV...")

    debug = False

    temp = create_header_nodes_to_csv(proc_name, output_file_name)

    writer = temp[0]
    out = temp[1]

    results = write_body_nodes_to_csv_func(wn, node_names, list_results, writer, debug)

    tot_nodes_demand = results[0]
    tot_leak_demand = results[1]

    write_bottom_nodes_to_csv_func(wn, out, tot_nodes_demand, tot_leak_demand, output_file_name, proc_name, number_of_nodes_with_leaks, number_of_nodes_with_sensors)

if __name__ == "__main__":

    input_file_inp1 = "./networks/exported_month_large_complete_one_reservoirs_small.inp"
    # input_file_inp2 = "./networks/exported_month_large_complete_one_reservoirs_large.inp"
    # input_file_inp3 = "./networks/exported_month_large_complete_two_reservoirs.inp"

    number_of_junctions_with_leaks1 = 0
    # number_of_junctions_with_leaks2 = 0
    # number_of_junctions_with_leaks3 = 0

    number_of_nodes_with_sensors1 = 0
    # number_of_nodes_with_sensors2 = 0
    # number_of_nodes_with_sensors3 = 0

    # sim_duration = 744 * 3600
    sim_duration = 24 * 3600 #TODO: 200 hours simulation

    wn_1 = wntr.network.WaterNetworkModel(input_file_inp1)

    # wn_2 = wntr.network.WaterNetworkModel(input_file_inp2)
    # wn_2.options.time.duration = 744 * 3600
    #
    # wn_3 = wntr.network.WaterNetworkModel(input_file_inp3)
    # wn_3.options.time.duration = 744 * 3600

    smart_sensor_junctions = []

    smart_sensor_junctions1 = []
    # smart_sensor_junctions2 = []
    # smart_sensor_junctions3 = []

    if (smart_sensors_enabled):
        print("SMART SENSORS ENABLED")

        number_of_nodes_with_sensors1 = int(len(wn_1.junction_name_list) / 4)
        # number_of_nodes_with_sensors2 = int(len(wn_2.junction_name_list) / 4)
        # number_of_nodes_with_sensors3 = int(len(wn_3.junction_name_list) / 4)

        smart_sensor_junctions1 = pick_rand_smart_sensors(wn_1, number_of_nodes_with_sensors1)
        # smart_sensor_junctions2 = pick_rand_smart_sensors(wn_2,number_of_nodes_with_sensors2)
        # smart_sensor_junctions3 = pick_rand_smart_sensors(wn_3,number_of_nodes_with_sensors3)
    else:
        print("SMART SENSORS NOT ENABLED")

    print("FIRST RUN WITHOUT LEAKS")

    list_of_demands = generate_rand_demand_array(0, 0.005, wn_1.num_junctions, (sim_duration / 3600))

    run_sim_with_random_base_demands(wn_1, sim_duration, smart_sensor_junctions1,
                                     output_file_name="1D_one_res_small_no_leaks_rand_base_dem_",
                                     proc_name="1D_one_res_small_no_leaks_rand_base_dem: ",
                                     number_of_nodes_with_sensors=number_of_nodes_with_sensors1,
                                     number_of_nodes_with_leaks=number_of_junctions_with_leaks1,
                                     list_of_demands=list_of_demands)

    wn_1.reset_initial_values()

    leaks_enabled = True


    if (leaks_enabled):
        print("LEAKAGES ENABLED")

        number_of_junctions_with_leaks1 = int(len(wn_1.junction_name_list) / 2)
        # number_of_junctions_with_leaks2 = int(len(wn_2.junction_name_list) / 2)
        # number_of_junctions_with_leaks3 = int(len(wn_3.junction_name_list) / 2)

        selected_junctions1 = pick_rand_leaks(wn_1,number_of_junctions_with_leaks1)
        # selected_junctions2 = pick_rand_leaks(wn_2,number_of_junctions_with_leaks2)
        # selected_junctions3 = pick_rand_leaks(wn_3,number_of_junctions_with_leaks3)

        assign_leaks(wn_1, 0.0002, selected_junctions1, "1D_one_res_small")
        # assign_leaks(wn_2, 0.000006, selected_junctions2, "1D_one_res_large")
        # assign_leaks(wn_3, 0.000006, selected_junctions3, "1D_two_res_large")
    else:
        print("LEAKAGES NOT ENABLED")

    #Code to be ran with a single execution

    # run_sim(wn_1, sim_duration, smart_sensor_junctions1, output_file_name="1M_one_res_small_no_leaks_same_base_dem_",
    #         proc_name="1M_one_res_small_no_leaks_same_base_dem: ",
    #         number_of_nodes_with_sensors=number_of_nodes_with_sensors1,
    #         number_of_nodes_with_leaks=number_of_junctions_with_leaks1)
    #
    # wn_1.reset_initial_values()

    run_sim_with_random_base_demands(wn_1, sim_duration, smart_sensor_junctions1,
                                     output_file_name="1D_ALT_one_res_small_with_leaks_rand_base_dem_",
                                     proc_name="1D_ALT_one_res_small_with_leaks_rand_base_dem: ",
                                     number_of_nodes_with_sensors=number_of_nodes_with_sensors1,
                                     number_of_nodes_with_leaks=number_of_junctions_with_leaks1,
                                     list_of_demands=list_of_demands)

    # run_sim(wn_2, smart_sensor_junctions2, output_file_name="1M_one_res_large_alt_with_leaks_", proc_name="1M_one_res_large_alt_with_leaks: ",
    #         number_of_nodes_with_sensors= number_of_nodes_with_sensors2, number_of_nodes_with_leaks=number_of_junctions_with_leaks2)
    #
    # run_sim(wn_3, smart_sensor_junctions3, output_file_name="1M_two_res_large_alt_with_leaks_", proc_name="1M_two_res_large_alt_with_leaks: ",
    #         number_of_nodes_with_sensors= number_of_nodes_with_sensors3, number_of_nodes_with_leaks=number_of_junctions_with_leaks3)

    #Code to be ran with multiple execution (useful for producing parallel multiple leaks)
    #proc1 = WNTR_Process("1D_one_res_small", wn_1, smart_sensor_junctions1)
    #proc2 = WNTR_Process("1D_one_res_large", wn_2, smart_sensor_junctions2)
    #proc3 = WNTR_Process("1D_two_res_large", wn_3, smart_sensor_junctions3)

    #proc1.start()
    #proc2.start()
    #proc3.start()

    #proc1.join()
    #proc2.join()
    #proc3.join()

    print("Exiting...")