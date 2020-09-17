"""
Plots script for JOCN 2020 paper.
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
    list_folders = ['C (96)', 'C+L (192)', 'C+L+S (288)', 'C+L+S (384)', 'C+L+S (384) Flat']

    # Creates dictionary of curves (marker/color)
    color_dict = {
        list_folders[0]: 'b',
        list_folders[1]: 'r',
        list_folders[2]: 'g',
        list_folders[3]: 'orange',
        list_folders[4]: 'k'
    }
    marker_dict = {
        list_folders[0]: 'o',
        list_folders[1]: 'o',
        list_folders[2]: 'o',
        list_folders[3]: 'o',
        list_folders[4]: 'o'
    }
    # Markers/line width configurations
    marker_size = 1.5
    thin_line = 1
    thick_line = 2
    handles = []

    # Frequencies
    freq_range_c = [191.30e12, 196.05e12]
    freq_range_l = [186.05e12, 190.80e12]
    freq_range_s = [196.55e12, 206.10e12]

    # Figure initial parameters
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize=(8, 5))

    # C-band only
    name = list_folders[0]
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_c96 = [hz2thz(freq) for freq in data.frequencies]
    gsnr_c96 = [lin2db(gsnr) for gsnr in data.gsnr]
    h, = plt.plot(freq_c96, gsnr_c96, '.', marker=marker_dict[name], c=color_dict[name], markersize=marker_size,
                  label=name)
    handles.append(copy.copy(h))
    txt_file.write(name + '\n')
    txt_file.write('C band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_c96), np.min(gsnr_c96), np.max(gsnr_c96)))

    # C+L
    name = list_folders[1]
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_c96l96 = [hz2thz(freq) for freq in data.frequencies]
    gsnr_c96l96 = [lin2db(gsnr) for gsnr in data.gsnr]
    gsnr_c96l96_c = [gsnr_c96l96[i] for i, freq in enumerate(data.frequencies) if
                     freq_range_c[0] <= freq <= freq_range_c[1]]
    gsnr_c96l96_l = [gsnr_c96l96[i] for i, freq in enumerate(data.frequencies) if
                     freq_range_l[0] <= freq <= freq_range_l[1]]
    h, = plt.plot(freq_c96l96, gsnr_c96l96, '.', marker=marker_dict[name], c=color_dict[name], markersize=marker_size,
                  label=name)
    handles.append(copy.copy(h))
    txt_file.write(name + '\n')
    txt_file.write('L band: {}\t{}\t{}\n'.format(np.mean(gsnr_c96l96_l), np.min(gsnr_c96l96_l), np.max(gsnr_c96l96_l)))
    txt_file.write('C band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_c96l96_c), np.min(gsnr_c96l96_c),
                                                   np.max(gsnr_c96l96_c)))

    # C+L+S (96-S)
    name = list_folders[2]
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_c96l96s96 = [hz2thz(freq) for freq in data.frequencies]
    gsnr_c96l96s96 = [lin2db(gsnr) for gsnr in data.gsnr]
    gsnr_c96l96s96_c = [gsnr_c96l96s96[i] for i, freq in enumerate(data.frequencies) if
                        freq_range_c[0] <= freq <= freq_range_c[1]]
    gsnr_c96l96s96_l = [gsnr_c96l96s96[i] for i, freq in enumerate(data.frequencies) if
                        freq_range_l[0] <= freq <= freq_range_l[1]]
    gsnr_c96l96s96_s = [gsnr_c96l96s96[i] for i, freq in enumerate(data.frequencies) if
                        freq_range_s[0] <= freq <= freq_range_s[1]]
    h, = plt.plot(freq_c96l96s96, gsnr_c96l96s96, '.', marker=marker_dict[name], c=color_dict[name],
                  markersize=marker_size, label=name)
    handles.append(copy.copy(h))
    txt_file.write(name + '\n')
    txt_file.write('L band: {}\t{}\t{}\n'.format(np.mean(gsnr_c96l96s96_l), np.min(gsnr_c96l96s96_l),
                                                 np.max(gsnr_c96l96s96_l)))
    txt_file.write('C band: {}\t{}\t{}\n'.format(np.mean(gsnr_c96l96s96_c), np.min(gsnr_c96l96s96_c),
                                                 np.max(gsnr_c96l96s96_c)))
    txt_file.write('S band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_c96l96s96_s), np.min(gsnr_c96l96s96_s),
                                                   np.max(gsnr_c96l96s96_s)))

    # C+L+S (192-S)
    name = list_folders[3]
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_c96l96s192 = [hz2thz(freq) for freq in data.frequencies]
    gsnr_c96l96s192 = [lin2db(gsnr) for gsnr in data.gsnr]
    freq_c = [hz2thz(freq) for freq in data.frequencies if freq_range_c[0] <= freq <= freq_range_c[1]]
    freq_l = [hz2thz(freq) for freq in data.frequencies if freq_range_l[0] <= freq <= freq_range_l[1]]
    freq_s = [hz2thz(freq) for freq in data.frequencies if freq_range_s[0] <= freq <= freq_range_s[1]]
    gsnr_c96l96s192_c = [gsnr_c96l96s192[i] for i, freq in enumerate(data.frequencies) if
                         freq_range_c[0] <= freq <= freq_range_c[1]]
    gsnr_c96l96s192_l = [gsnr_c96l96s192[i] for i, freq in enumerate(data.frequencies) if
                         freq_range_l[0] <= freq <= freq_range_l[1]]
    gsnr_c96l96s192_s = [gsnr_c96l96s192[i] for i, freq in enumerate(data.frequencies) if
                         freq_range_s[0] <= freq <= freq_range_s[1]]
    h, = plt.plot(freq_c96l96s192, gsnr_c96l96s192, '.', marker=marker_dict[name], c=color_dict[name],
                  markersize=marker_size, label=name)
    handles.append(copy.copy(h))
    txt_file.write(name + '\n')
    txt_file.write('L band: {}\t{}\t{}\n'.format(np.mean(gsnr_c96l96s192_l), np.min(gsnr_c96l96s192_l),
                                                 np.max(gsnr_c96l96s192_l)))
    txt_file.write('C band: {}\t{}\t{}\n'.format(np.mean(gsnr_c96l96s192_c), np.min(gsnr_c96l96s192_c),
                                                 np.max(gsnr_c96l96s192_c)))
    txt_file.write('S band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_c96l96s192_s), np.min(gsnr_c96l96s192_s),
                                                   np.max(gsnr_c96l96s192_s)))

    plt.hlines(y=np.mean(gsnr_c96l96s192_s), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]),
               linestyles='dashdot', colors=color_dict[name], linewidth=thin_line)
    plt.hlines(y=np.min(gsnr_c96l96s192_s), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]),
               linestyles='solid', colors=color_dict[name], linewidth=thin_line)
    plt.hlines(y=np.max(gsnr_c96l96s192_s), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]),
               linestyles='solid', colors=color_dict[name], linewidth=thin_line)

    # C+L+S (192-S) Flat
    name = list_folders[4]
    mat_file = [file for file in os.listdir(prof_folder / name) if file.endswith('.mat')][0]
    config = json.load(open(prof_folder / (name + '/' + 'config_file.json'), 'r'))
    data = DataQot.load_qot_mat(prof_folder / (name + '/' + mat_file))
    freq_c96l96s192 = [hz2thz(freq) for freq in data.frequencies]
    gsnr_c96l96s192 = [lin2db(gsnr) for gsnr in data.gsnr]
    freq_c = [hz2thz(freq) for freq in data.frequencies if freq_range_c[0] <= freq <= freq_range_c[1]]
    freq_l = [hz2thz(freq) for freq in data.frequencies if freq_range_l[0] <= freq <= freq_range_l[1]]
    freq_s = [hz2thz(freq) for freq in data.frequencies if freq_range_s[0] <= freq <= freq_range_s[1]]
    gsnr_c96l96s192_c = [gsnr_c96l96s192[i] for i, freq in enumerate(data.frequencies) if
                         freq_range_c[0] <= freq <= freq_range_c[1]]
    gsnr_c96l96s192_l = [gsnr_c96l96s192[i] for i, freq in enumerate(data.frequencies) if
                         freq_range_l[0] <= freq <= freq_range_l[1]]
    gsnr_c96l96s192_s = [gsnr_c96l96s192[i] for i, freq in enumerate(data.frequencies) if
                         freq_range_s[0] <= freq <= freq_range_s[1]]
    h, = plt.plot(freq_c96l96s192, gsnr_c96l96s192, '.', marker=marker_dict[name], c=color_dict[name],
                  markersize=marker_size, label=name)
    handles.append(copy.copy(h))
    txt_file.write(name + '\n')
    txt_file.write('L band: {}\t{}\t{}\n'.format(np.mean(gsnr_c96l96s192_l), np.min(gsnr_c96l96s192_l),
                                                 np.max(gsnr_c96l96s192_l)))
    txt_file.write('C band: {}\t{}\t{}\n'.format(np.mean(gsnr_c96l96s192_c), np.min(gsnr_c96l96s192_c),
                                                 np.max(gsnr_c96l96s192_c)))
    txt_file.write('S band: {}\t{}\t{}\n\n'.format(np.mean(gsnr_c96l96s192_s), np.min(gsnr_c96l96s192_s),
                                                   np.max(gsnr_c96l96s192_s)))

    plt.hlines(y=np.mean(gsnr_c96l96s192_s), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]),
               linestyles='dashdot', colors=color_dict[name], linewidth=thin_line)
    plt.hlines(y=np.min(gsnr_c96l96s192_s), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]),
               linestyles='solid', colors=color_dict[name], linewidth=thin_line)
    plt.hlines(y=np.max(gsnr_c96l96s192_s), xmin=hz2thz(freq_range_s[0]), xmax=hz2thz(freq_range_s[1]),
               linestyles='solid', colors=color_dict[name], linewidth=thin_line)

    # Lines or colors to define the bands
    plt.axvline(x=hz2thz(freq_range_c[0]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_c[1]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_l[0]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_l[1]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_s[0]), color='k', linewidth=thin_line)
    plt.axvline(x=hz2thz(freq_range_s[1]), color='k', linewidth=thin_line)
    # plt.axvspan(hz2thz(freq_range_c[0]), hz2thz(freq_range_c[1]), facecolor='orange', alpha=0.25)
    # plt.axvspan(hz2thz(freq_range_l[0]), hz2thz(freq_range_l[1]), facecolor='red', alpha=0.25)
    # plt.axvspan(hz2thz(freq_range_s[0]), hz2thz(freq_range_s[1]), facecolor='yellow', alpha=0.25)

    # Bands text boxes
    high = 28
    ax.text((np.mean(freq_c)), high, 'C-Band', fontsize=18, color='k', horizontalalignment='center',
            verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='k'))
    ax.text((np.mean(freq_l)), high, 'L-Band', fontsize=18, color='k', horizontalalignment='center',
            verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='k'))
    ax.text((np.mean(freq_s)), high, 'S-Band', fontsize=18, color='k', horizontalalignment='center',
            verticalalignment='bottom', bbox=dict(facecolor='white', edgecolor='k'))

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
    leg = plt.legend(loc="lower left", handles=handles, fontsize=16, markerscale=3)
    plt.tight_layout()

    # Save and/or plot figures
    if save_figs:
        plt.savefig((output_folder / 'GSNR.pdf'), bbox_inches='tight', dpi=600)
        plt.savefig((output_folder / 'GSNR.eps'), bbox_inches='tight', dpi=300, format='eps')
    if plot_figs:
        plt.show()


def plot_traffic(data_path, topology, traffic_type, save_figs=True, plot_figs=False, smooth_plot=True):
    prof_folder = data_path / 'Network_Assessment/{}/{}'.format(topology, traffic_type)
    output_folder = data_path / 'Figures'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # List of folders
    list_folders = ['C (96)', 'SDM (192)', 'SDM (288)', 'SDM (384)', 'BDM C+L (192)', 'BDM C+L+S (288)',
                    'BDM C+L+S (384)']

    # Y axis limits (BP)
    y_min = 1e-4
    y_max = 1e-1

    # Curves colors and format
    format_dict = {
        list_folders[0]: 'k',
        list_folders[1]: 'r-',
        list_folders[2]: 'r--',
        list_folders[3]: 'r-.',
        list_folders[4]: 'g-',
        list_folders[5]: 'g--',
        list_folders[6]: 'g-.',
    }

    # Load all data curves
    list_curves = []
    for folder in list_folders:
        mat_file = [file for file in os.listdir(prof_folder / folder) if file.endswith('.mat')][0]
        mat_path = (prof_folder / (folder + '/' + mat_file))
        data = DataTraffic.load_traffic_mat(path=mat_path, name=folder)
        list_curves.append(data)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["legend.loc"] = "lower right"

    # Configuration for allocated traffic
    plt.figure(figsize=(8, 4))
    for curve in list_curves:
        if smooth_plot:
            x, y = utils.get_interval(curve.total_acc_traffic, curve.prob_rejected, y_min, y_max)
            x, y = utils.smooth_curve(x, y, poly=3, div=2.0)
            plt.plot(x, y, format_dict[curve.name], label=curve.name, linewidth=1.5)
        else:
            plt.plot(curve.total_acc_traffic, curve.prob_rejected, label=curve.name, linewidth=2)

    # Final plot configuration
    plt.ylabel('Blocking probability', fontsize=18, fontweight='bold')
    plt.xlabel('Network Total Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.yscale('log')
    plt.ylim(y_min, y_max)
    plt.xlim(200, 1800)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # leg = plt.legend(fontsize=14)
    # for leg_obj in leg.legendHandles:
    #     leg_obj.set_linewidth(2.0)
    plt.grid(b=True, which='minor', linestyle='-.', linewidth=0.5)
    plt.grid(b=True, which='major', linestyle='-', linewidth=1.0)
    plt.tight_layout()

    # Save and/or plot figures
    if save_figs:
        plt.savefig((output_folder / '{}_{}_AllocTraffic_BP.pdf'.format(topology, traffic_type)), bbox_inches='tight',
                    dpi=600)
        plt.savefig((output_folder / '{}_{}_AllocTraffic_BP.eps'.format(topology, traffic_type)), bbox_inches='tight',
                    dpi=300, format='eps')
    if plot_figs:
        plt.show()


def plot_multi_factor(data_path, topology, save_figs=True, plot_figs=False, bp_thr=1e-2):
    prof_folder = data_path / 'Network_Assessment/{}'.format(topology)
    output_folder = data_path / 'Figures'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    txt_file = open(output_folder / 'Alloc_traffic.txt', 'w')
    txt_file.write('BP threshold: {}\n'.format(bp_thr))

    # List of folders
    list_folders = ['C (96)', 'SDM (192)', 'BDM C+L (192)', 'SDM (288)', 'BDM C+L+S (288)', 'SDM (384)',
                    'BDM C+L+S (384)']
    # Names of x axis
    names = ['C only', 'SDM 2x', 'BDM CL', 'SDM 3x', 'BDM CLS(96)', 'SDM 4x', 'BDM CLS(192)']
    x = np.arange(len(names))

    # Color pallet for legend
    color_dict = {
        'Uniform': 'r',
        'Nonuniform': 'g'
    }

    # Loads all data files
    bars_traffic = {}
    for traffic_type in os.listdir(prof_folder):
        aux_bar = []
        for folder in list_folders:
            mat_file = [file for file in os.listdir(prof_folder / (traffic_type + '/' + folder))
                        if file.endswith('.mat')][0]
            mat_path = (prof_folder / (traffic_type + '/' + folder + '/' + mat_file))
            data = DataTraffic.load_traffic_mat(path=mat_path, name=folder, thr_bp=bp_thr)
            aux_bar.append(data)
        bars_traffic[traffic_type] = aux_bar

    # Calculate the multiplicative factor for each curve
    values_dict = {}
    for key in bars_traffic.keys():
        base_value = [data.alloc_traffic for data in bars_traffic[key] if data.name == list_folders[0]][0]
        values_list, multi_list = [], []
        txt_file.write('Scenario: {}\n'.format(key))
        for data in bars_traffic[key]:
            data.calc_multi_factor(base_traffic=base_value)
            values_list.append(data.alloc_traffic)
            multi_list.append(data.multi_factor)
            txt_file.write('Allocated traffic for {} case: {}\n'.format(data.name, data.alloc_traffic))
        values_dict[key] = (values_list, multi_list)
        txt_file.write('\n')

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    bar_width = 0.4  # set the width of the bars
    opacity = 1.0    # not so dark

    # Plot bars
    aux_var = 0
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in values_dict.keys():
        plt.bar(x + aux_var, values_dict[key][0], width=bar_width, color=color_dict[key], alpha=opacity, zorder=1,
                label=key)
        for i, name in enumerate(list_folders):
            if i > 0:
                ax.text(x[i] + aux_var, values_dict[key][0][i], str(values_dict[key][1][i]),
                        horizontalalignment='center', verticalalignment='bottom', fontsize=16, color='k')
        aux_var += bar_width

    # Final parameters
    plt.grid(b=False, which='major', axis='x')
    plt.ylabel('Total Alloc. Traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.ylim(0, 1700)
    plt.xlim()
    plt.xticks(x + (bar_width / 2), names, fontsize=14, rotation=35, ha='right')
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc="upper left")
    plt.tight_layout()

    # Save and/or plot figures
    if save_figs:
        plt.savefig((output_folder / '{}_Multi_factor.pdf'.format(topology)), bbox_inches='tight', dpi=600)
        plt.savefig((output_folder / '{}_Multi_factor.eps'.format(topology)), bbox_inches='tight', dpi=300,
                    format='eps')
    if plot_figs:
        plt.show()


if __name__ == "__main__":
    path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                'Publications_Data/JOCN_Power_Optimization')

    # Plot for GSNR profile (Uncomment to run)
    # plot_gsnr(data_path=path)

    # Plot for allocated traffic
    top = 'DT'
    traffic = 'Nonuniform'
    # plot_traffic(data_path=path, topology=top, traffic_type=traffic)

    # Plot multiplicative factor
    # plot_multi_factor(data_path=path, topology=top)
