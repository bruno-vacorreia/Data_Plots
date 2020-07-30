import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path


def bp_lightpaths(snap_param, simul_path, title=False, plot=False, save=False):
    topology_name = snap_param['topology']
    data_path = simul_path / 'BPvLightpaths.csv'
    data = pd.read_csv(filepath_or_buffer=data_path, sep='\t', lineterminator='\n')
    aux_list = [value for i, value in enumerate(data['Num_Lightpaths']) if data['BP'].loc[i] > 0.0]
    min_value = min(aux_list)
    max_value = max(aux_list)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig, ax = plt.subplots()
    if title:
        plt.title('Allocated lightpaths versus call request BP for {} topology'.format(topology_name))
    ax.plot(data['Num_Lightpaths'], data['BP'], 'r-', lw=2, label='plot1')
    plt.xlabel('Allocated lightpaths', fontsize=18, fontweight='bold')
    plt.xlim((min_value - 10), (max_value + 10))
    plt.yscale('log')
    plt.ylabel('Blocking probability', fontsize=18, fontweight='bold')
    plt.tight_layout()

    if save:
        fig_path = simul_path / 'Figures'
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        plt.savefig(fig_path / 'BPvLightpaths_{}.png'.format(topology_name), bbox_inches='tight')
        plt.savefig(fig_path / 'BPvLightpaths_{}.pdf'.format(topology_name), bbox_inches='tight')

    if plot:
        plt.show()


def topology_congestion(snap_param, simul_path, title=False, plot=False, save=False):
    topology_name = snap_param['topology']
    data_path = simul_path / 'Congestion.csv'
    data = pd.read_csv(filepath_or_buffer=data_path, sep='\t', lineterminator='\n')
    bar_width = 0.6

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig, ax = plt.subplots()
    if title:
        plt.title('Congestion for {} topology'.format(topology_name))
    ax.barh(data['Fibers'], data['Congestion'], bar_width, color='blue', zorder=2)
    ax.xaxis.set_major_formatter(PercentFormatter())
    plt.yticks(ha='right', fontsize=8)
    plt.xticks(np.linspace(0, 100, 11))
    plt.xlabel('Congestion', fontsize=18, fontweight='bold')
    plt.tight_layout()

    if save:
        fig_path = simul_path / 'Figures'
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        plt.savefig(fig_path / 'Congestion_{}.png'.format(topology_name), bbox_inches='tight')
        plt.savefig(fig_path / 'Congestion_{}.pdf'.format(topology_name), bbox_inches='tight')

    if plot:
        plt.show()


if __name__ == "__main__":
    SIMULATION_PATH = Path('')
    param = json.load(open(SIMULATION_PATH / 'config_file.json'))
    snap_params = param['snap_params']

    bp_lightpaths(snap_params, SIMULATION_PATH, title=False, plot=True, save=True)
    topology_congestion(snap_params, SIMULATION_PATH, title=False, plot=True, save=True)






