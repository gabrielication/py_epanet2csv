import multiprocessing

import wntr
import csv
import sys
from decimal import Decimal
import random

from multiprocessing import Process, Lock

smart_sensors_enabled = True
leaks_enabled = True

lock = Lock()

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

def run_sim(wn, smart_sensor_junctions, output_file_name="", proc_name=""):
    print(proc_name + "Configuring simulation...")

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    node_names = wn.node_name_list
    link_names = wn.link_name_list

    sim_duration_in_seconds = wn.options.time.duration

    #print(dict(wn.options.hydraulic))

    #results = wntr.sim.EpanetSimulator(wn).run_sim()
    print(proc_name + "Running simulation...")

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    print(proc_name + "Simulation finished. Writing to csv (can take a while)...")

    nodes_to_csv(wn, results, node_names, output_file_name, proc_name, smart_sensor_junctions)
    links_to_csv(wn, results, link_names, output_file_name, proc_name)

    print(proc_name + "Finished!")

def pick_rand_leaks(wn):
    node_names = wn.junction_name_list

    #selected_junctions = random.sample(node_names, 3)

    len_nodes = int(len(node_names) / 2)
    selected_junctions = random.sample(node_names, len_nodes)

    print(len_nodes)
    print(len(selected_junctions))

    return selected_junctions

def assign_leaks(wn, area_size, selected_junctions, proc_name):
    for node_id in selected_junctions:
        node_obj = wn.get_node(node_id)

        node_obj.add_leak(wn, area=area_size, start_time=0)

        print(proc_name + "Leak added to node id: ", node_id)

def pick_rand_smart_sensors(wn):
    node_names = wn.junction_name_list

    len_nodes = int(len(node_names) / 4)
    selected_junctions = random.sample(node_names, len_nodes)

    print(len_nodes)
    print(len(selected_junctions))

    return selected_junctions

def links_to_csv(wn, results, link_names, output_file_name, proc_name):
    print(proc_name + "Writing Links' CSV...")

    flow_results = results.link['flowrate']
    velocity_results = results.link['velocity']
    #headloss_results = results.link['headloss'] NOT COMPATIBLE WITH WNTR

    indexes = flow_results.index

    outName = output_file_name+"links_output.csv"
    out = open(outName, "w")
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

def nodes_to_csv(wn, results, node_names, output_file_name, proc_name, smart_sensor_junctions):
    print(proc_name + "Writing Nodes' CSV...")

    demand_results = results.node['demand']
    head_results = results.node['head']
    pressure_results = results.node['pressure']

    indexes = demand_results.index

    outName = output_file_name+"nodes_output.csv"
    out = open(outName, "w")
    writer = csv.writer(out)

    debug = False

    for timestamp in indexes:
        tot_demand = Decimal('0')
        demand_value = Decimal('0')

        #hour = pd.to_datetime(timestamp, unit='s').time()

        hour = str(int(timestamp/3600))+":00:00" #cheap way to calculate timestamps. if we choose an interval != 1h then it will break

        #print(hour)

        for nodeID in node_names:
            node_obj = wn.get_node(nodeID)

            demand_value = Decimal(str(demand_results.loc[timestamp, nodeID]))
            demand_value = round(demand_value,8)
            head_value = Decimal(str(head_results.loc[timestamp, nodeID]))
            head_value = round(head_value,8)
            pressure_value = Decimal(str(pressure_results.loc[timestamp, nodeID]))
            pressure_value = round(pressure_value,8)

            smart_sensor_is_present = 0 #can be 0,1,2

            #has_leak = node_obj._leak #this leak-flag represents if a leak was set to the node but not if the leak is flowing now on this timestamp
            has_leak = False

            leak_area_value = node_obj.leak_area #I think that this does not require an approximation... right?

            leak_discharge_value = node_obj.leak_discharge_coeff #Same as above...?

            current_leak_demand_value = results.node["leak_demand"].at[timestamp, nodeID]
            '''
            if(current_leak_demand_value > 0.0):
                has_leak = True #this leak-flag is set to true if the leak is flowing now on this timestamp
            '''
            if (leak_area_value > 0.0):
                has_leak = True  # this leak-flag is set to true if we see a hole in the node

            if(smart_sensors_enabled):
                if(nodeID in smart_sensor_junctions):
                    if(has_leak):
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

            tot_demand = tot_demand + demand_value
            tot_demand = round(tot_demand,8)

            if debug:
                print("tot_d")
                print("{:4.15f}".format(tot_demand))

            if debug:
                if nodeID=="101":
                    print("test")
                    sys.exit(1)

            x_pos = node_obj.coordinates[0]
            y_pos = node_obj.coordinates[1]

            node_type = node_obj.__class__.__name__

            output_row = [hour, nodeID, demand_value, head_value, pressure_value, x_pos, y_pos,
                          node_type, has_leak, leak_area_value, leak_discharge_value, current_leak_demand_value, smart_sensor_is_present]

            # print(nodeID)
            # print(demand_value)
            # print(tot_demand)
            # print("Node (" + str(nodeID) + ") demand is: "+str(demand_value)+" so Tot demand is now: " + str(tot_demand))
            # print("Node ({}) demand is: {} so Tot demand is now: {}".format(nodeID,demand_value,tot_demand))

            writer.writerow(output_row)

        # print(timestamp)
        if debug:
            if tot_demand<1e-6:
                print("Tot demand at "+str(hour)+" is: 0 ("+str(tot_demand)+")")
            else:
                print("Tot demand at "+str(hour)+" is: "+str(tot_demand))
        # break

    out.close()

    print(proc_name + "Nodes' CSV written.")

if __name__ == "__main__":

    input_file_inp1 = "./networks/exported_month_large_complete_one_reservoirs_small.inp"
    input_file_inp2 = "./networks/exported_month_large_complete_one_reservoirs_large.inp"
    input_file_inp3 = "./networks/exported_month_large_complete_two_reservoirs.inp"

    #Code to be ran with a single execution

    wn_1 = wntr.network.WaterNetworkModel(input_file_inp1)
    wn_1.options.time.duration = 24 * 3600

    wn_2 = wntr.network.WaterNetworkModel(input_file_inp2)
    wn_2.options.time.duration = 24 * 3600

    wn_3 = wntr.network.WaterNetworkModel(input_file_inp3)
    wn_3.options.time.duration = 24 * 3600

    smart_sensor_junctions = []

    if (leaks_enabled):
        selected_junctions1 = pick_rand_leaks(wn_1)
        selected_junctions2 = pick_rand_leaks(wn_2)
        selected_junctions3 = pick_rand_leaks(wn_3)

        assign_leaks(wn_1, 0.05, selected_junctions1, "1D_one_res_small")
        assign_leaks(wn_2, 0.05, selected_junctions2, "1D_one_res_large")
        assign_leaks(wn_3, 0.05, selected_junctions3, "1D_two_res_large")

    if (smart_sensors_enabled):
        smart_sensor_junctions1 = pick_rand_smart_sensors(wn_1)
        smart_sensor_junctions2 = pick_rand_smart_sensors(wn_2)
        smart_sensor_junctions3 = pick_rand_smart_sensors(wn_3)

    #Code to be ran with multiple execution (useful for producing parallel multiple leaks)
    proc1 = WNTR_Process("1D_one_res_small", wn_1, smart_sensor_junctions1)
    proc2 = WNTR_Process("1D_one_res_large", wn_2, smart_sensor_junctions2)
    proc3 = WNTR_Process("1D_two_res_large", wn_3, smart_sensor_junctions3)

    proc1.start()
    proc2.start()
    proc3.start()

    proc1.join()
    proc2.join()
    proc3.join()

    print("Exiting...")