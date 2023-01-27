import wntr
import csv
import sys
from decimal import Decimal
import random

import pandas as pd

def write_results_to_csv_with_pandas(results, filename):
    # Create an empty DataFrame

    nodes = results.node

    df = pd.DataFrame(columns=list(nodes.keys()))
    df.loc[0] = list(nodes.values())

    print(df)

def run_sim(wn, sim_duration, number_of_nodes_with_leaks=0, output_file_name=""):
    print("Configuring simulation...")

    wn.options.hydraulic.demand_model = 'PDD' #Pressure Driven Demand mode

    # wn.options.hydraulic.required_pressure = 21.097  # 30 psi = 21.097 m
    # wn.options.hydraulic.minimum_pressure = 3.516  # 5 psi = 3.516 m

    wn.options.time.duration = sim_duration

    print("Running simulation...")

    results = wntr.sim.WNTRSimulator(wn).run_sim()

    # pressure = results[0].node['pressure']
    #
    # for node_id in node_names:
    #     print(node_id,pressure.loc[3600, node_id])

    write_results_to_csv_with_pandas(results, "to_csv.csv")

    print("Simulation finished...")

if __name__ == "__main__":

    input_file_inp1 = "./networks/exported_month_large_complete_one_reservoirs_small.inp"

    wn_1 = wntr.network.WaterNetworkModel(input_file_inp1)
    sim_duration = 24 * 3600

    run_sim(wn_1, sim_duration)