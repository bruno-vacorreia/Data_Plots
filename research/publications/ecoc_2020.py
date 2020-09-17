"""
Plots script for ECOC 2020 paper.
"""

import os
import json
import copy
import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from gnpy.core.utils import lin2db
from research.power_optimization.utils.utils import hz2thz
from research.power_optimization.data import DataTraffic, DataQot
import research.power_optimization.utils.utils as utils

# Plot color configuration
sns.set(color_codes=True)
sns.set(palette="deep", font_scale=1.1, color_codes=True, rc={"figure.figsize": [8, 5]})


def plot_gsnr(data_path, save_figs=True, plot_figs=False):
    # Get name of folders
    prof_folder = data_path / 'Profiles'
    output_folder = data_path / 'Figures'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    txt_file = open(output_folder / 'values_GSNR.txt', 'w')
    txt_file.write('GSNR average / minimum GSNR / maximum GSNR\n\n')

    # List of folders
    list_folders = ['C only', 'BDM', 'BDM Optimized']

    # Creates dictionary of curves (marker/color)
    color_dict = {
        list_folders[0]: 'b',
        list_folders[1]: 'r',
        list_folders[2]: 'g'
    }
    marker_dict = {
        list_folders[0]: 'o',
        list_folders[1]: 'o',
        list_folders[2]: 'o'
    }
    marker_size = 3
    thin_line = 1
    thick_line = 2
    handles = []

    # Figure initial parameters
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["legend.loc"] = "lower left"
    fig, ax = plt.subplots(figsize=(8, 5))

    # # Frequencies
    freq_range_c = [191.30e12, 196.05e12]
    freq_range_l = [186.05e12, 190.80e12]
    freq_range_s = [196.55e12, 206.10e12]

    # C band simulation
    name = 'C only'
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_c = [hz2thz(freq) for freq in data.frequencies]
    gsnr_c = [lin2db(gsnr) for gsnr in data.gsnr]

    h, = plt.plot(freq_c, gsnr_c, '.', marker=marker_dict[name], c=color_dict[name], markersize=marker_size, label=name)
    handles.append(copy.copy(h))
    plt.axvline(x=hz2thz(freq_range_c[0]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_c[1]), color='k', linewidth=thin_line)
    txt_file.write('{}\n'.format(name))
    txt_file.write('C band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_c), np.min(gsnr_c), np.max(gsnr_c)))

    # C+L+S simulations
    name = 'BDM'
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_cls_1 = [hz2thz(freq) for freq in data.frequencies]
    gsnr_cls_1 = [lin2db(gsnr) for gsnr in data.gsnr]
    freq_l = [hz2thz(freq) for freq in data.frequencies if freq_range_l[0] <= freq <= freq_range_l[1]]
    freq_s = [hz2thz(freq) for freq in data.frequencies if freq_range_s[0] <= freq <= freq_range_s[1]]
    gsnr_c_1 = [gsnr_cls_1[i] for i, freq in enumerate(data.frequencies) if freq_range_c[0] <= freq <= freq_range_c[1]]
    gsnr_l_1 = [gsnr_cls_1[i] for i, freq in enumerate(data.frequencies) if freq_range_l[0] <= freq <= freq_range_l[1]]
    gsnr_s_1 = [gsnr_cls_1[i] for i, freq in enumerate(data.frequencies) if freq_range_s[0] <= freq <= freq_range_s[1]]
    txt_file.write('{}\n'.format(name))
    txt_file.write('L band: {}\t{}\t{}\n'.format(np.mean(gsnr_l_1), np.min(gsnr_l_1), np.max(gsnr_l_1)))
    txt_file.write('C band: {}\t{}\t{}\n'.format(np.mean(gsnr_c_1), np.min(gsnr_c_1), np.max(gsnr_c_1)))
    txt_file.write('S band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_s_1), np.min(gsnr_s_1), np.max(gsnr_s_1)))

    h, = plt.plot(freq_cls_1, gsnr_cls_1, '.', marker=marker_dict[name], c=color_dict[name], markersize=marker_size,
                  label=name)
    handles.append(copy.copy(h))
    plt.axvline(x=hz2thz(freq_range_l[0]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_l[1]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_s[0]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_s[1]), color='k', linewidth=thin_line)
    plt.hlines(y=np.mean(gsnr_s_1), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]), linestyles='dashdot',
               colors=color_dict[name], linewidth=thick_line)
    plt.hlines(y=np.min(gsnr_s_1), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]), linestyles='solid',
               colors=color_dict[name], linewidth=thick_line)
    plt.hlines(y=np.max(gsnr_s_1), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]), linestyles='solid',
               colors=color_dict[name], linewidth=thick_line)

    name = 'BDM Optimized'
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_cls_2 = [hz2thz(freq) for freq in data.frequencies]
    gsnr_cls_2 = [lin2db(gsnr) for gsnr in data.gsnr]
    gsnr_c_2 = [gsnr_cls_2[i] for i, freq in enumerate(data.frequencies) if freq_range_c[0] <= freq <= freq_range_c[1]]
    gsnr_l_2 = [gsnr_cls_2[i] for i, freq in enumerate(data.frequencies) if freq_range_l[0] <= freq <= freq_range_l[1]]
    gsnr_s_2 = [gsnr_cls_2[i] for i, freq in enumerate(data.frequencies) if freq_range_s[0] <= freq <= freq_range_s[1]]
    txt_file.write('{}\n'.format(name))
    txt_file.write('L band: {}\t{}\t{}\n'.format(np.mean(gsnr_l_2), np.min(gsnr_l_2), np.max(gsnr_l_2)))
    txt_file.write('C band: {}\t{}\t{}\n'.format(np.mean(gsnr_c_2), np.min(gsnr_c_2), np.max(gsnr_c_2)))
    txt_file.write('S band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_s_2), np.min(gsnr_s_2), np.max(gsnr_s_2)))

    h, = plt.plot(freq_cls_2, gsnr_cls_2, '.', marker=marker_dict[name], c=color_dict[name], markersize=marker_size,
                  label=name)
    handles.append(copy.copy(h))
    # plt.axvline(x=hz2thz(freq_range_l[0]), color='k')
    # plt.axvline(x=hz2thz(freq_range_l[1]), color='k')
    # plt.axvline(x=hz2thz(freq_range_s[0]), color='k')
    # plt.axvline(x=hz2thz(freq_range_s[1]), color='k')
    plt.hlines(y=np.mean(gsnr_s_2), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]), linestyles='dashdot',
               colors=color_dict[name], linewidth=thick_line)
    plt.hlines(y=np.min(gsnr_s_2), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]), linestyles='solid',
               colors=color_dict[name], linewidth=thick_line)
    plt.hlines(y=np.max(gsnr_s_2), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]), linestyles='solid',
               colors=color_dict[name], linewidth=thick_line)

    # Bands legends
    ax.text((np.mean(freq_c)) - 1.6, 28.5, 'C-Band', fontsize=18, color='k',
            bbox=dict(facecolor='white', edgecolor='k'))
    ax.text((np.mean(freq_l)) - 1.6, 28.5, 'L-Band', fontsize=18, color='k',
            bbox=dict(facecolor='white', edgecolor='k'))
    ax.text((np.mean(freq_s)) - 1.6, 28.5, 'S-Band', fontsize=18, color='k',
            bbox=dict(facecolor='white', edgecolor='k'))

    # Final parameters
    plt.ylabel('GSNR [dB]', fontsize=18, fontweight='bold')
    plt.xlabel('Frequency [THz]', fontsize=18, fontweight='bold')
    plt.ylim(23, 32)
    plt.xlim(185.0, 207.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Put all legend markets in black
    # for h in handles:
    #     h.set_color("black")
    leg = plt.legend(handles=handles, fontsize=16, markerscale=2)

    plt.tight_layout()
    if save_figs:
        plt.savefig((output_folder / 'GSNR.pdf'), bbox_inches='tight', dpi=600)

    if plot_figs:
        plt.show()


def plot_traffic(data_path, save_figs=True, plot_figs=False, smooth_plot=False):
    # Get name of folders
    prof_folder = data_path / 'Network_Assessment'
    output_folder = data_path / 'Figures'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # List of folders
    list_folders = ['BDM C+L+S', 'SDM (4x fibers)']

    # Y axis limits (BP)
    y_min = 1e-4
    y_max = 1e-1

    # Save all curves for a topology/Traffic
    list_curves = []
    for sub_folder in list_folders:
        mat_file = [file for file in os.listdir(prof_folder / sub_folder) if file.endswith('.mat')]
        mat_file = mat_file[0]
        mat_path = (prof_folder / (sub_folder + '/' + mat_file))
        data = DataTraffic.load_traffic_mat(path=mat_path, name=sub_folder)
        list_curves.append(data)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["legend.loc"] = "lower right"

    format_dict = {
        list_folders[0]: 'r-',
        list_folders[1]: 'g-'
    }

    # Configuration for allocated traffic
    plt.figure(figsize=(8, 4))
    for curve in list_curves:
        if smooth_plot:
            x, y = utils.get_interval(curve.total_acc_traffic, curve.prob_rejected, y_min, y_max)
            x, y = utils.smooth_curve(x, y, poly=3, div=3.0)
            plt.plot(x, y, format_dict[curve.name], label=curve.name, linewidth=2.0)
        else:
            plt.plot(curve.total_acc_traffic, curve.prob_rejected, label=curve.name, linewidth=2)

    # Final parameters
    plt.ylabel('Blocking probability', fontsize=18, fontweight='bold')
    plt.xlabel('Total Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.yscale('log')
    plt.ylim(y_min, y_max)
    plt.xlim(1000, 1310)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.grid(b=True, which='minor', linestyle='-.', linewidth=0.5)
    plt.grid(b=True, which='major', linestyle='-', linewidth=1.0)
    plt.tight_layout()

    # Save and/or plot figures
    if save_figs:
        plt.savefig((output_folder / 'AllocTraffic_BP.pdf'), bbox_inches='tight', dpi=600)
    if plot_figs:
        plt.show()


def plot_multi_factor(data_path, save_figs=True, plot_figs=False, bp_thr=1e-2):
    # Get name of folders
    prof_folder = data_path / 'Network_Assessment'
    output_folder = data_path / 'Figures'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    txt_file = open(output_folder / 'Alloc_traffic.txt', 'w')
    txt_file.write('BP threshold: {}\n'.format(bp_thr))

    # List of folders
    list_folders = ['C only', 'BDM C+L+S', 'SDM (4x fibers)']
    x = np.arange(len(list_folders))

    # Save all curves for a topology/Traffic
    list_curves = []
    for sub_folder in list_folders:
        mat_file = [file for file in os.listdir(prof_folder / sub_folder) if file.endswith('.mat')]
        mat_file = mat_file[0]
        mat_path = (prof_folder / (sub_folder + '/' + mat_file))
        data = DataTraffic.load_traffic_mat(path=mat_path, name=sub_folder, thr_bp=bp_thr)
        list_curves.append(data)

    # Calculate the multiplicative factor for each curve
    values_list, multi_list = [], []
    base_value = [data.alloc_traffic for data in list_curves if data.name == list_folders[0]][0]
    for curve in list_curves:
        curve.calc_multi_factor(base_value)
        values_list.append(curve.alloc_traffic)
        multi_list.append(curve.multi_factor)
        txt_file.write('Allocated traffic for {} case: {}\n'.format(curve.name, curve.alloc_traffic))
    txt_file.write('\n')

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    bar_width = 0.3  # set the width of the bars
    opacity = 1.0  # not so dark

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.bar(x, values_list, width=bar_width, color='black', alpha=opacity, zorder=1)
    for i, name in enumerate(list_folders):
        if i > 0:
            ax.text(x[i], values_list[i], str(multi_list[i]), horizontalalignment='center', verticalalignment='bottom',
                    fontsize=18, color='k')

    # Final parameters
    plt.grid(b=False, which='major', axis='x')
    plt.ylabel('Total Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.ylim(0, 1300)
    plt.xlim()
    plt.xticks(x, list_folders, fontsize=18)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    # Save and/or plot figures
    if save_figs:
        plt.savefig((output_folder / 'Multi_factor.pdf'), bbox_inches='tight', dpi=600)
    if plot_figs:
        plt.show()


if __name__ == "__main__":
    path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                'Publications_Data/ECOC_Power_Optimization')

    # plot_gsnr(data_path=path, save_figs=True, plot_figs=False)
    # plot_traffic(data_path=path, save_figs=True, plot_figs=False, smooth_plot=True)
    # plot_multiplicative_factor(path, save_figs=True, plot_figs=False)
