import os
import json
import copy
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from pathlib import Path
from gnpy.core.utils import lin2db
import research.power_optimization.utils.utils as utils
from research.power_optimization.data import DataTraffic, DataQot


# Plot color configuration
sns.set(color_codes=True)
sns.set(palette="deep", font_scale=1.1, color_codes=True, rc={"figure.figsize": [8, 5]})


def compare_data():
    """
    Function to compare the data of OSI/GNPy with the data of internal Raman solver used by Emanuele.
    """
    default_path = '/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations/Results/' \
                   'Compare_Results/'
    gsnr_data = default_path + 'Old_Results/GSNR/C+L/GSNR/ofc_c+l__offset_c=1.0/offset_l=0.5/' \
                               'gsnr__ofc_c+l__offset_c=1.0_offset_l=0.5__tilt_c=-0.3_tilt_l=0.1.mat'
    raman_data = default_path + 'Old_Results/GSNR/C+L/RAMAN/ofc_c+l__offset_c=1.0/offset_l=0.5/' \
                                'ofc_c+l__offset_c=1.0_offset_l=0.5__tilt_c=-0.3_tilt_l=0.1.mat'
    new_sim = default_path + 'jocn_2020_l_c_offset_l=0.5_tilt_l=0.1_offset_c=1.0_tilt_c=-0.3.mat'
    figures_path = Path(__file__).parent / 'results/JOCN_2020/Figures'
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)

    gsnr_data = loadmat(gsnr_data)
    raman_data = loadmat(raman_data)
    new_sim = loadmat(new_sim)

    save_figures = True
    plot_gsnr = True
    plot_raman_profile = False
    plot_gain = True
    plot_ase = True

    # GSNR, SNR_nl and OSNR
    if plot_gsnr:
        plt.figure(1)
        plt.plot(utils.hz2thz(new_sim['frequencies'][0]), lin2db(new_sim['GSNR'][0]), 'b.', label='GSNR')
        plt.plot(utils.hz2thz(new_sim['frequencies'][0]), lin2db(new_sim['SNR_NL'][0]), 'g.', label='SNR_nl')
        plt.plot(utils.hz2thz(new_sim['frequencies'][0]), lin2db(new_sim['OSNR'][0]), 'r.', label='OSNR')
        plt.title('C+L+S - Single span', fontsize=20)
        plt.ylabel('Power ratio (dB)', fontsize=14)
        plt.xlabel('Frequencies (GHz)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save_figures:
            plt.savefig((figures_path / 'Single_span.png'), bbox_inches='tight')

        plt.figure(2)
        plt.plot(raman_data['f_channel'] / 1e12, 10 * np.log10(gsnr_data['GSNR']), 'b.', label='GSNR')
        plt.plot(raman_data['f_channel'] / 1e12, 10 * np.log10(gsnr_data['SNR_NL']), 'g.', label='SNR_nl')
        plt.plot(raman_data['f_channel'] / 1e12, 10 * np.log10(gsnr_data['OSNR']), 'r.', label='OSNR')
        plt.title('C+L - Single span (OFC/OMDM)', fontsize=20)
        plt.ylabel('Power ratio (dB)', fontsize=14)
        plt.xlabel('Frequencies (GHz)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save_figures:
            plt.savefig((figures_path / 'Single_span_OFC_OMDM.png'), bbox_inches='tight')

    # Gain and noise profile
    if plot_raman_profile:
        fig = plt.figure(3)
        ax = fig.gca(projection='3d')
        x = new_sim['f_axis'][0] / 1e12
        y = new_sim['z_ase'][0] / 1e3
        x, y = np.meshgrid(x, y)
        z = np.transpose(10 * np.log10(new_sim['raman_power']))
        surf = ax.plot_surface(x, y, z, linewidth=0, antialiased=False)
        plt.title('Raman power profile', fontsize=20)
        ax.set_xlabel('Frequencies (GHz)')
        ax.set_ylabel('Propagation direction (km)')
        ax.set_zlabel('Power (dBm)')

        fig = plt.figure(4)
        ax = fig.gca(projection='3d')
        x = raman_data['f_channel'] / 1e12
        y = raman_data['z_array'] / 1e3
        x, y = np.meshgrid(x, y)
        z = np.transpose(10 * np.log10(raman_data['raman_power']))
        surf2 = ax.plot_surface(x, y, z, linewidth=0, antialiased=False)
        plt.title('Raman power profile (OFC/OMDM)', fontsize=20)
        ax.set_xlabel('Frequencies (GHz)')
        ax.set_ylabel('Propagation direction (km)')
        ax.set_zlabel('Power (dBm)')

    if plot_gain:
        plt.figure(5)
        plt.plot(new_sim['frequencies'][0] / 1e12, new_sim['G'][0], 'b.', label='Gain')
        plt.plot(raman_data['f_channel'] / 1e12, gsnr_data['G'], 'g.', label='Gain OFC/OMDM')
        plt.title('C+L+S gain profile', fontsize=20)
        plt.ylabel('Gain (dB)', fontsize=14)
        plt.xlabel('Frequencies (GHz)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save_figures:
            plt.savefig((figures_path / 'Gain_profile.png'), bbox_inches='tight')

    if plot_ase:
        plt.figure(6)
        plt.plot(new_sim['frequencies'][0] / 1e12, 10 * np.log10(new_sim['ase'][0]), 'b.', label='ase')
        plt.plot(raman_data['f_axis'] / 1e12, 10 * np.log10(gsnr_data['ase']), 'g.', label='ase OFC/OMDM')
        plt.title('C+L+S ase power (Single span)', fontsize=20)
        plt.ylabel('Power (dBm)', fontsize=14)
        plt.xlabel('Frequencies (GHz)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        if save_figures:
            plt.savefig((figures_path / 'Ase.png'), bbox_inches='tight')

    plt.show()


def plot_freq_gsnr(data_path, name=None, figure_path=None, save_fig=False, plot_fig=False):
    data = DataQot.load_qot_mat(data_path)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.figure()
    plt.plot(utils.hz2thz(data.frequencies), lin2db(data.gsnr), 'b.', label='GSNR')
    plt.plot(utils.hz2thz(data.frequencies), lin2db(data.snr_nl), 'g.', label='SNR_nl')
    plt.plot(utils.hz2thz(data.frequencies), lin2db(data.osnr), 'r.', label='OSNR')

    plt.ylabel('Power ratio (dB)', fontsize=14)
    plt.xlabel('Frequencies (THz)', fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()

    if save_fig and figure_path:
        if not os.path.isdir(figure_path):
            os.makedirs(figure_path)
        if not name:
            name = data.name
        plt.savefig((figure_path / '{}.pdf'.format(name)), bbox_inches='tight')

    if plot_fig:
        plt.show()

    plt.clf()
    plt.close()


def plot_freq_powers(data_path, name=None, figure_path=None, save_fig=False, plot_fig=False):
    data = DataQot.load_qot_mat(data_path)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.figure()
    plt.plot(utils.hz2thz(data.frequencies), utils.lin2dbm(data.powers), 'b.', label='Powers')

    plt.ylabel('Power (dBm)', fontsize=14)
    plt.xlabel('Frequencies (THz)', fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()

    if save_fig and figure_path:
        if not os.path.isdir(figure_path):
            os.makedirs(figure_path)
        if not name:
            name = data.name
        plt.savefig((figure_path / '{}.pdf'.format(name)), bbox_inches='tight')

    if plot_fig:
        plt.show()

    plt.clf()
    plt.close()


def plot_best_combinations(data_path, best_combinations, output_folder=None, save_figs=False, plot_figs=True):
    if not output_folder:
        output_folder = data_path
    figures_path = output_folder / 'Figures/GSNR_Profiles'
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)

    for best in best_combinations:
        name_file = ''
        if isinstance(best_combinations, dict):
            name_file = best_combinations[best]
        elif isinstance(best_combinations, list):
            name_file = best
        plot_freq_gsnr((data_path / (name_file + '.mat')), name_file, figure_path=figures_path, save_fig=save_figs,
                       plot_fig=plot_figs)


def compute_pareto_front(save_figs=False, plot_figs=True):
    root_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Data_combinations/Data_processed/JOCN_2020')
    config_path = root_path / 'config_file.json'
    pareto_folder = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/research/Simulations_Data/Results/'
                         'JOCN_Power_Optimization/C_L_S/Pareto_front')
    if not os.path.isdir(pareto_folder):
        os.mkdir(pareto_folder)

    list_data_files = [file for file in os.listdir(root_path) if file.endswith('.mat')]
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']

    best_gsnr = 0.0
    best_power = np.inf
    population = []
    names = []

    # print('Computing the best GSNR(largest GSNR average) and best power(minimum average power)')
    # list_data_files = tqdm(iterable=list_data_files, desc='Computing GSNR and power')
    # for file in list_data_files:
    #     data = loadmat(root_path / file)
    #     comb_name = str(data['name'][0])
    #     names.append(comb_name)
    #     channels_indices = data['cut_index'][0]
    #     calc_gsnr_db = [lin2db(gsnr) for i, gsnr in enumerate(data['GSNR'][0]) if i in channels_indices]
    #     calc_power_dbm = [lin2db(power) for i, power in enumerate(data['powers'][0]) if i in channels_indices]
    #     calc_power = [power for i, power in enumerate(data['powers'][0]) if i in channels_indices]
    #     average_gsnr = np.mean(calc_gsnr_db)
    #     average_power = np.mean(calc_power)
    #
    #     delta_gsnr_band = []
    #     num_ch_band = 0
    #     for band in bands:
    #         calc_gsnr_band = [gsnr for gsnr in calc_gsnr_db[num_ch_band:(num_ch_band +
    #                                                                      len(bands[band]['comp_channels']))]]
    #         # calc_gsnr_band = [gsnr for gsnr in calc_gsnr_db[num_ch_band:(num_ch_band + bands[band]['nb_channels'])]]
    #         max_gsnr_band = max(calc_gsnr_band)
    #         min_gsnr_band = min(calc_gsnr_band)
    #         delta_gsnr_band.append(max_gsnr_band - min_gsnr_band)
    #         num_ch_band += len(bands[band]['comp_channels'])
    #
    #     population.append([average_gsnr, 1 / np.mean(delta_gsnr_band)])

    # population = np.array(population)
    # np.save(pareto_folder / 'Pareto.npy', population)
    population = np.load(pareto_folder / 'Pareto.npy')

    pareto = utils.identify_pareto_max(population)
    print('Indices of non-dominated solutions:\n{}'.format(pareto))
    pop_pareto = population[pareto]
    # names_pareto = [names[index] for index in pareto]
    #
    # with open((pareto_folder / 'Pareto_names.txt'), 'w') as pareto_file:
    #     for name in names_pareto:
    #         pareto_file.write('{}\n'.format(name))

    pop_pareto_df = pd.DataFrame(pop_pareto)
    pop_pareto_df.sort_values(0, inplace=True)
    pop_pareto = pop_pareto_df.values

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.plot(1 / population[:, 1], population[:, 0], 'b.')
    plt.plot(1 / pop_pareto[:, 1], pop_pareto[:, 0], color='r')
    plt.xlabel('Average GSNR variation [dB]', fontsize=18)
    plt.ylabel('GSNR [dB]', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0.48)

    # Plot inside (Zoom)
    sub_axes = plt.axes([.63, .63, .25, .25], facecolor='lightgray')
    sub_axes.plot(1 / population[:, 1], population[:, 0], 'b.')
    sub_axes.plot(1 / pop_pareto[:, 1], pop_pareto[:, 0], color='r')
    plt.xlim(0.4, 0.8)
    plt.ylim(28.6, 29.15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if save_figs:
        plt.savefig((pareto_folder / '{}.png'.format('Combinations')), bbox_inches='tight')
        plt.savefig((pareto_folder / '{}.pdf'.format('Combinations')), bbox_inches='tight')
    if plot_figs:
        plt.show()


def plot_alloc_traffic(path_data, plot_figs=True, save_figs=False, smooth_plot=False):
    # Y axis limits (BP)
    y_min = 1e-4
    y_max = 1e-1

    # Dictionary
    format_dict = {
        'Reference C Band (1x fiber)': 'k',
        'BDM C+L Band (1x fiber)': 'b-',
        'BDM C+L+S Band (1x fiber)': 'b--',
        'SDM-CCC C Band (2x fibers)': 'g-',
        'SDM-CCC C Band (3x fibers)': 'g--',
        'SDM-InS C Band (2x fibers)': 'r-',
        'SDM-InS C Band (3x fibers)': 'r--'
    }

    # for folder in list_folders:
    list_curves = []
    topology_traffic_path = path_data / 'Figures'
    if not os.path.isdir(topology_traffic_path):
        os.makedirs(topology_traffic_path)
    list_sub_folders = [file for file in os.listdir(path_data)
                        if os.path.isdir(path_data / file)]
    index_ref = [i for i, file in enumerate(list_sub_folders) if 'Reference' in file][0]
    ref_folder = list_sub_folders[index_ref]
    list_sub_folders.remove(ref_folder)
    list_sub_folders.insert(0, ref_folder)

    # Save all curves for a topology/Traffic
    for sub_folder in list_sub_folders:
        mat_file = [file for file in os.listdir(path_data / sub_folder) if file.endswith('.mat')]
        if mat_file:
            mat_path = (path_data / (sub_folder + '/' + mat_file[0]))
            data = DataTraffic.load_traffic_mat(path=mat_path, name=sub_folder)
            list_curves.append(data)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Configuration for allocated traffic
    plt.figure(figsize=(8, 4))
    for curve in list_curves:
        if smooth_plot:
            x, y = utils.get_interval(curve.total_acc_traffic, curve.prob_rejected, y_min, y_max)
            x, y = utils.smooth_curve(x, y, poly=3, div=2.0)
            plt.plot(x, y, format_dict[curve.name], label=curve.name, linewidth=2.0)
        else:
            plt.plot(curve.total_acc_traffic, curve.prob_rejected, label=curve.name, linewidth=2)
    plt.ylabel('Blocking probability', fontsize=18, fontweight='bold')
    plt.xlabel('Total Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.yscale('log')
    plt.ylim(y_min, y_max)
    plt.xlim(200, 1500)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(fontsize=11)
    plt.grid(b=True, which='minor', linestyle='-.', linewidth=0.5)
    plt.grid(b=True, which='major', linestyle='-', linewidth=1.0)
    plt.tight_layout()
    if save_figs:
        plt.savefig((topology_traffic_path / 'AllocTraffic_BP.pdf'), bbox_inches='tight', dpi=600)
    if plot_figs:
        plt.show()


def plot_alloc_traffic_optimization(plot_figs=True, save_figs=False):
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Future_scenarios_analyze/Allocated_traffic')

    # for folder in list_folders:
    list_curves = []
    list_sub_folders = [file for file in os.listdir(path_data) if os.path.isdir(path_data / file)]

    # Save all curves for a topology/Traffic
    for sub_folder in list_sub_folders:
        mat_file = [file for file in os.listdir(path_data / sub_folder) if file.endswith('.mat')]
        mat_file = mat_file[0]
        mat_path = (path_data / (sub_folder + '/' + mat_file))
        data = DataTraffic.load_traffic_mat(path=mat_path, name=sub_folder)
        list_curves.append(data)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Configuration for allocated traffic
    plt.figure(figsize=(8, 4))
    for curve in list_curves:
        plt.plot(curve.total_acc_traffic, curve.prob_rejected, label=curve.name, linewidth=2)
    plt.ylabel('Blocking probability', fontsize=18, fontweight='bold')
    plt.xlabel('Total Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.yscale('log')
    plt.ylim(1e-4, 2e-1)
    plt.xlim(200, 1300)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(b=True, which='minor', linestyle='-.')
    plt.grid(b=True, which='major', linestyle='-')
    plt.tight_layout()
    if save_figs:
        plt.savefig((path_data / 'AllocTraffic_BP.png'), bbox_inches='tight', dpi=600)
        plt.savefig((path_data / 'AllocTraffic_BP.pdf'), bbox_inches='tight', dpi=600)
    if plot_figs:
        plt.show()


def plot_gsnr(plot_figs=True, save_figs=False):
    """
    Plot of GSNR for channels computed per band, loading from a .npy file
    """
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Compare_GSNR')
    handles = []
    color_c = 'b'
    color_l = 'r'
    color_s = 'g'
    marker_c = 'o'
    marker_cl = 'v'
    marker_cls = 's'
    marker_size = 8

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot for C band only
    name = 'C'
    data_c = np.load(path_data / (name + '.npy'))
    freq_c = data_c[0]
    freq_c_begin = 191.30e12
    freq_c_end = 196.10e12
    gsnr_c = data_c[1]

    # Lines for C-band
    h, = plt.plot(utils.hz2thz(freq_c), gsnr_c, '.', marker=marker_c, c=color_c, markersize=marker_size,
                  label=(name + ' Band GSNR'))
    handles.append(copy.copy(h))
    plt.axvline(x=utils.hz2thz(freq_c_begin), color=color_c)
    plt.axvline(x=utils.hz2thz(freq_c_end), color=color_c)
    plt.hlines(y=np.mean(gsnr_c), xmin=utils.hz2thz(freq_c_begin), xmax=utils.hz2thz(freq_c_end), linestyles='dashdot',
               colors=color_c)

    # Plot for C+L
    name = 'C+L'
    data_cl = np.load(path_data / (name + '.npy'))
    freq_cl = data_cl[0]
    freq_l = np.array([freq for freq in freq_cl if freq not in freq_c])
    freq_l_begin = 186.05e12
    freq_l_end = 190.8e12
    gsnr_l_cl = [gsnr for i, gsnr in enumerate(data_cl[1]) if i < len(freq_c)]
    gsnr_c_cl = [gsnr for i, gsnr in enumerate(data_cl[1]) if i >= len(freq_c)]

    h, = plt.plot(utils.hz2thz(freq_l), gsnr_l_cl, '.', marker=marker_cl, c=color_l, markersize=marker_size,
                  label=(name + ' Band GSNR'))
    plt.plot(utils.hz2thz(freq_c), gsnr_c_cl, '.', marker=marker_cl, c=color_c, markersize=marker_size)
    # Lines for L-band
    handles.append(copy.copy(h))
    plt.axvline(x=utils.hz2thz(freq_l_begin), color=color_l)
    plt.axvline(x=utils.hz2thz(freq_l_end), color=color_l)
    plt.hlines(y=np.mean(gsnr_c_cl), xmin=utils.hz2thz(freq_c_begin), xmax=utils.hz2thz(freq_c_end),
               linestyles='dotted', colors=color_c)
    plt.hlines(y=np.mean(gsnr_l_cl), xmin=utils.hz2thz(freq_l_begin), xmax=utils.hz2thz(freq_l_end),
               linestyles='dotted', colors=color_l)

    # Plot for C+L+S
    name = 'C+L+S'
    data_cls = np.load(path_data / (name + '.npy'))
    freq_s = np.array([freq for freq in data_cls[0] if freq not in freq_cl])
    freq_s_begin = 196.55e12
    freq_s_end = 201.3e12
    gsnr_l_cls = [gsnr for i, gsnr in enumerate(data_cls[1]) if i < len(freq_c)]
    gsnr_c_cls = [gsnr for i, gsnr in enumerate(data_cls[1]) if len(freq_c) <= i < (2 * len(freq_c))]
    gsnr_s_cls = [gsnr for i, gsnr in enumerate(data_cls[1]) if i >= (2 * len(freq_c))]

    h, = plt.plot(utils.hz2thz(freq_l), gsnr_l_cls, '.', marker=marker_cls, c=color_l, markersize=marker_size,
                  label=(name + ' Band GSNR'))
    plt.plot(utils.hz2thz(freq_c), gsnr_c_cls, '.', marker=marker_cls, c=color_c, markersize=marker_size)
    plt.plot(utils.hz2thz(freq_s), gsnr_s_cls, '.', marker=marker_cls, c=color_s, markersize=marker_size)
    # Lines for S-band
    handles.append(copy.copy(h))
    plt.axvline(x=utils.hz2thz(freq_s_begin), color=color_s)
    plt.axvline(x=freq_s_end / 1e12, color=color_s)
    plt.hlines(y=np.mean(gsnr_c_cls), xmin=utils.hz2thz(freq_c_begin), xmax=utils.hz2thz(freq_c_end),
               linestyles='solid', colors=color_c)
    plt.hlines(y=np.mean(gsnr_l_cls), xmin=utils.hz2thz(freq_l_begin), xmax=utils.hz2thz(freq_l_end),
               linestyles='solid', colors=color_l)
    plt.hlines(y=np.mean(gsnr_s_cls), xmin=utils.hz2thz(freq_s_begin), xmax=utils.hz2thz(freq_s_end),
               linestyles='solid', colors=color_s)

    ax.text((utils.hz2thz(np.mean(freq_c))) - 1.2, 28, 'C-Band', fontsize=18, color=color_c,
            bbox=dict(facecolor='white', edgecolor=color_c))
    ax.text((utils.hz2thz(np.mean(freq_l))) - 1.2, 28, 'L-Band', fontsize=18, color=color_l,
            bbox=dict(facecolor='white', edgecolor=color_l))
    ax.text((utils.hz2thz(np.mean(freq_s))) - 1.2, 28, 'S-Band', fontsize=18, color=color_s,
            bbox=dict(facecolor='white', edgecolor=color_s))

    plt.ylabel('GSNR [dB]', fontsize=18, fontweight='bold')
    plt.xlabel('Frequency [THz]', fontsize=18, fontweight='bold')
    plt.ylim()
    plt.xlim()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Put all legend markets in black
    for h in handles:
        h.set_color("black")
    leg = plt.legend(handles=handles, fontsize=18)

    plt.tight_layout()
    if save_figs:
        plt.savefig((path_data / 'GSNR.png'), bbox_inches='tight', dpi=600)
        plt.savefig((path_data / 'GSNR.pdf'), bbox_inches='tight', dpi=600)

    if plot_figs:
        plt.show()


def plot_noise_figure_gain(plot_fig=True, save_fig=False):
    data_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Noise_Figure_S_band')
    gain_path = data_path / 'gain'
    noise_path = data_path / 'noise_figures'

    marker_c = 'o'
    marker_l = 'v'
    marker_s = 's'

    # Read data for S band
    data_s_band = pd.read_csv(filepath_or_buffer=(noise_path / 's_band_noise_figure.csv'), memory_map=True)
    freqs_s_band = np.array(data_s_band['frequencies'])
    noise_figure_s_band = np.array(data_s_band['noise_figure'])
    # # Read data for C band
    # data_c_band = pd.read_csv(filepath_or_buffer=(noise_path / 'c_band_noise_figure.csv'), memory_map=True)
    # freqs_c_band = np.array(data_c_band['frequencies'])
    # noise_figure_c_band = np.array(data_c_band['noise_figure'])
    # # Read data for L band
    # data_l_band = pd.read_csv(filepath_or_buffer=(noise_path / 'l_band_noise_figure.csv'), memory_map=True)
    # freqs_l_band = np.array(data_l_band['frequencies'])
    # noise_figure_l_band = np.array(data_l_band['noise_figure'])

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # general_plots
    # plt.plot(freqs_l_band / 1e12, noise_figure_l_band, '.', marker=marker_c, markersize=8, color='red',
    #          label='Noise figure for L-band')
    # plt.plot(freqs_c_band / 1e12, noise_figure_c_band, '.', marker=marker_c, markersize=8, color='blue',
    #          label='Noise figure for C-band')
    plt.plot(utils.hz2thz(freqs_s_band), noise_figure_s_band, '.', marker=marker_c, markersize=8, color='green',
             label='Noise figure for S-band')

    # Plot configurations
    plt.ylabel('Noise figure [dB]', fontsize=18, fontweight='bold')
    plt.xlabel('Frequency [THz]', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim()
    plt.ylim(5, 10.5)
    plt.legend(fontsize=18)

    plt.savefig((data_path / '{}.png'.format('Noise_fig')), bbox_inches='tight')
    plt.savefig((data_path / '{}.pdf'.format('Noise_fig')), bbox_inches='tight')

    plt.show()


def plot_traffic(bp_thr=1e-2, plot_fig=True, save_fig=False):
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Future_scenarios_analyze/Allocated_traffic')

    list_folders = [file for file in os.listdir(path_data) if os.path.isdir(path_data / file)]

    plot = {}
    for folder in list_folders:
        file = [file for file in os.listdir(path_data / folder) if file.endswith('.mat')][0]
        mat_data = loadmat(path_data / (folder + '/' + file))
        average_accept_req = np.transpose(mat_data['cell_averageCumAcceptDemands'][0][0])
        norm_traffic_band = np.transpose(mat_data['cell_norm_traffic_band'][0][0])
        norm_traffic_lambda = np.transpose(mat_data['cell_norm_traffic_lambda'][0][0])
        prob_reject = np.transpose(mat_data['cell_probReject'][0][0])
        total_acc_traffic = np.transpose(mat_data['cell_totalAcceptedTraffic'][0][0])

        index = 0
        for i, bp in enumerate(prob_reject):
            if bp >= bp_thr:
                index = i
                break
        plot[folder] = {'Alloc_traffic': total_acc_traffic[index][0]}

    opt_c, opt_cl, opt_cls = [], [], []
    for key in plot.keys():
        if '(Opt C)' in key:
            opt_c.append(plot[key]['Alloc_traffic'])
        elif '(Opt C+L)' in key:
            opt_cl.append(plot[key]['Alloc_traffic'])
        elif '(Opt C+L+S)' in key:
            opt_cls.append(plot[key]['Alloc_traffic'])
    opt_c.sort()
    opt_cl.sort()
    opt_cls.sort()
    x_labels = ['C', 'C+L', 'C+L+S']
    x = np.arange(len(x_labels))

    print('Opt C:')
    for tra in opt_c:
        print('Allocated traffic [Tbps] for BP={}: {}'.format(bp_thr, tra))

    print('Opt C+L:')
    for tra in opt_cl:
        print('Allocated traffic [Tbps] for BP={}: {}'.format(bp_thr, tra))

    print('Opt C+L+S:')
    for tra in opt_cls:
        print('Allocated traffic [Tbps] for BP={}: {}'.format(bp_thr, tra))

    bar_width = 0.1
    opacity = 0.8
    desl = bar_width

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    plt.figure(figsize=(8, 4))
    plt.bar(x - desl, opt_c, width=bar_width, color='blue', label='Opt C', alpha=opacity)
    plt.bar(x, opt_cl, width=bar_width, color='orange', label='Opt C+L', alpha=opacity)
    plt.bar(x + desl, opt_cls, width=bar_width, color='green', label='Opt C+L+S', alpha=opacity)

    plt.grid(zorder=1, alpha=0.4, ls='dashed', axis='y', which='both')
    plt.legend()
    plt.xlabel('Spectral bands used', fontsize=18, fontweight='bold')
    plt.ylabel('Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.xticks(x, x_labels)
    plt.tight_layout()

    if save_fig:
        plt.savefig((path_data / 'Bands_AllocTraffic.png'), bbox_inches='tight', dpi=600)
        plt.savefig((path_data / 'Bands_AllocTraffic.pdf'), bbox_inches='tight', dpi=600)

    if plot_fig:
        plt.show()


if __name__ == "__main__":
    path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                'JOCN_Power_Optimization/C_L_S/Data_combinations/Data_processed_Sband_96_Regular/results/'
                'Allocated_traffic/Allocated_traffic_Plot/COST_ununiform')

    plot_alloc_traffic(path, plot_figs=False, save_figs=True, smooth_plot=True)
