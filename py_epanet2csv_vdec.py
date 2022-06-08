import multiprocessing

import wntr
import csv
import sys
from decimal import Decimal
import random

from multiprocessing import Process, Lock

lock = Lock()

class WNTR_Process (Process):
    def __init__(self, name, inp_file):
        Process.__init__(self)
        self.name = name
        self.inp_file = inp_file

    def run(self):
        print ("Proc '" + self.name + "' started")
        # Lock on
        lock.acquire()
        wn = wntr.network.WaterNetworkModel(self.inp_file)
        # Free lock
        lock.release()
        run_sim(wn, output_file_name=self.name+"_", proc_name=self.name + ": ")

def run_sim(wn, output_file_name="", proc_name=""):
    print(proc_name + "Configuring simulation...")

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    node_names = wn.node_name_list
    link_names = wn.link_name_list

    sim_duration_in_seconds = wn.options.time.duration

    leaks = True

    if(leaks):
        pick_three_rand_leaks(wn, 0.05, proc_name) #leakages start at half the duration (e.g. 1 year, start at 6 month)

    #print(dict(wn.options.hydraulic))

    #results = wntr.sim.EpanetSimulator(wn).run_sim()
    print(proc_name + "Running simulation...")

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    print(proc_name + "Simulation finished. Writing to csv (can take a while)...")

    nodes_to_csv(wn, results, node_names, output_file_name, proc_name)
    links_to_csv(wn, results, link_names, output_file_name, proc_name)

    print(proc_name + "Finished!")

def pick_three_rand_leaks(wn, area_size, proc_name):
    node_names = wn.junction_name_list

    selected_junctions = random.sample(node_names, 3)

    #len_nodes = int(len(node_names) / 2)
    #selected_junctions = random.sample(node_names, len_nodes)

    for node_id in selected_junctions:
        node_obj = wn.get_node(node_id)

        node_obj.add_leak(wn, area=area_size, start_time=0)

        print(proc_name + "Leak added to node id: ", node_id)

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

def nodes_to_csv(wn, results, node_names, output_file_name, proc_name):
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

            #has_leak = node_obj._leak #this leak-flag represents if a leak was set to the node but not if the leak is flowing now on this timestamp
            has_leak = False

            leak_area_value = node_obj.leak_area #I think that this does not require an approximation... right?

            leak_discharge_value = node_obj.leak_discharge_coeff #Same as above...?

            current_leak_demand_value = results.node["leak_demand"].at[timestamp, nodeID]

            if(current_leak_demand_value > 0.0):
                has_leak = True #this leak-flag is set to true if the leak is flowing now on this timestamp

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
                          node_type, has_leak, leak_area_value, leak_discharge_value, current_leak_demand_value]

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

    input_file_inp = "./networks/exported_month_large_complete_one_reservoirs_small.inp"

    #Code to be ran with a single execution
    '''
    wn = wntr.network.WaterNetworkModel(input_file_inp)
    run_sim(wn)
    '''

    #Code to be ran with multiple execution (useful for producing parallel multiple leaks)
    proc1 = WNTR_Process("P1", input_file_inp)
    proc2 = WNTR_Process("P2", input_file_inp)

    proc1.start()
    proc2.start()

    proc1.join()
    proc2.join()

    print("Exiting...")