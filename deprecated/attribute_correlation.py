#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quali features sono maggiormente correlate all'ETa?
1) Si usa la Scattermatrix per una valutazione visiva;
2) Si usano algoritmi di Feature Importance da Scikit learn.
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def plot_scattermatrix(df):
    sm = pd.plotting.scatter_matrix(df, alpha=0.6, figsize=(10, 10), diagonal='kde')

    # Ruota le etichette
    [s.xaxis.label.set_rotation(90) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]

    # Imposta la dimensione delle label
    [s.xaxis.label.set_fontsize(16) for s in sm.reshape(-1)]
    [s.yaxis.label.set_fontsize(16) for s in sm.reshape(-1)]

    # Sposta le etichette per non farle sovrapporre alla figura
    [s.get_yaxis().set_label_coords(-0.7, 0.35) for s in sm.reshape(-1)]

    # Nasconde i ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]

    plt.savefig('scatter_matrix.png')
    # plt.savefig('Images/scatter_matrix.eps')
    plt.show()

filename = 'sample_nodes_output_one_week.csv'

print("Loading csv...")

data = pd.read_csv(filename, names=["hour","nodeID","demand","head","pressure","x_pos", "y_pos",
                                    "node_type", "has_leak", "leak_area", "leak_discharge", "current_leak_demand"])

data["has_leak"] = data["has_leak"].astype(int)

#%% SCATTERMATRIX
# Si crea la matrice di correlazione tra le Features
matrix = plot_scattermatrix(data)
# .savefig('Images/scatter_matrix.png')
# # PLOTS
# fig = plt.figure(figsize = (16,9))
# plt.plot(df_SWC)
# plt.plot(mean_SWC)
# plt.legend(['θ 10', 'θ 20', 'θ 30', 'θ 40', 'θ 50', 'θ mean', ])
# plt.title('Soil Water [deficit]')
# plt.show()