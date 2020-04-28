from turtle import width

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    # Paths to files
    path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/UFPE/Pesquisa/Artigos/R-SA_SA-R')
    r_sa_path = path / 'LinksUse_NSFNET_RSA_L=264.txt'
    sa_r_path = path / 'LinksUse_NSFNET_SAR_L=264.txt'
    ga_path = path / 'LinksUse_NSFNET_HYB_L=264.txt'

    # Read data
    r_sa_data = pd.read_csv(filepath_or_buffer=r_sa_path, sep="\t", header=None, names=['Link', 'Use'])
    sa_r_data = pd.read_csv(filepath_or_buffer=sa_r_path, sep="\t", header=None, names=['Link', 'Use'])
    ga_data = pd.read_csv(filepath_or_buffer=ga_path, sep="\t", header=None, names=['Link', 'Use'])
    x_axis = np.arange(len(r_sa_data))
    r_sa_max_min = [max(r_sa_data.loc[slice(None), 'Use']), min(r_sa_data.loc[slice(None), 'Use'])]
    sa_r_max_min = [max(sa_r_data.loc[slice(None), 'Use']), min(sa_r_data.loc[slice(None), 'Use'])]
    ga_max_min = [max(ga_data.loc[slice(None), 'Use']), min(ga_data.loc[slice(None), 'Use'])]

    # Values for plot
    bar_width = 0.2
    bar_width_2 = 0.2
    desloc = 0.2
    opacity = 0.7
    opacity_2 = 0.4

    fig, ax = plt.subplots()
    ax.bar(x_axis, r_sa_data.loc[slice(None), 'Use'], label='R_SA', width=bar_width, color='blue', alpha=opacity)
    ax.bar(x_axis - desloc, sa_r_data.loc[slice(None), 'Use'], label='SA_R', width=bar_width, color='green',
            alpha=opacity)
    ax.bar(x_axis + desloc, ga_data.loc[slice(None), 'Use'], label='Hybrid', width=bar_width, color='red',
            alpha=opacity)
    ax.bar(min(x_axis) - 2, r_sa_max_min[0]-r_sa_max_min[1], bottom=r_sa_max_min[1], width=bar_width_2, color='blue')
    ax.bar(min(x_axis) - (2 + (bar_width_2*2)), sa_r_max_min[0] - sa_r_max_min[1], bottom=sa_r_max_min[1],
           width=bar_width_2, color='green')
    ax.bar(min(x_axis) - (2 - (bar_width_2 * 2)), ga_max_min[0] - ga_max_min[1], bottom=ga_max_min[1],
           width=bar_width_2, color='red')
    ax.plot([min(x_axis) - 2, max(x_axis)], [r_sa_max_min[1], r_sa_max_min[1]], "b--", lw=0.4, alpha=opacity_2)
    ax.plot([min(x_axis) - 2, max(x_axis)], [r_sa_max_min[0], r_sa_max_min[0]], "b--", lw=0.4, alpha=opacity_2)
    ax.plot([min(x_axis) - (2 + (bar_width_2*2)), max(x_axis)], [sa_r_max_min[1], sa_r_max_min[1]], "g--", lw=0.4,
            alpha=opacity_2)
    ax.plot([min(x_axis) - (2 + (bar_width_2*2)), max(x_axis)], [sa_r_max_min[0], sa_r_max_min[0]], "g--", lw=0.4,
            alpha=opacity_2)
    ax.plot([min(x_axis) - (2 - (bar_width_2 * 2)), max(x_axis)], [ga_max_min[1], ga_max_min[1]], "r--", lw=0.4,
            alpha=opacity_2)
    ax.plot([min(x_axis) - (2 - (bar_width_2 * 2)), max(x_axis)], [ga_max_min[0], ga_max_min[0]], "r--", lw=0.4,
            alpha=opacity_2)
    # plt.tight_layout()
    plt.xticks(x_axis, r_sa_data.loc[slice(None), 'Link'], fontsize=6, rotation=60, ha='right')
    plt.yticks(fontsize=6)
    plt.ylabel('Number of requests')
    plt.legend(fontsize=8)
    plt.title('Use of links')

    # Sav
    plt.savefig(path / 'NSFNet.png', bbox_inches='tight')
    plt.savefig(path / 'NSFNet.pdf', bbox_inches='tight')

    plt.show()
