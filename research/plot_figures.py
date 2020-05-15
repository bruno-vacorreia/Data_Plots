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
from research.utils.utils import identify_pareto_max, covnert_hz_thz

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
        plt.plot(new_sim['frequencies'][0] / 1e12, 10 * np.log10(new_sim['GSNR'][0]), 'b.', label='GSNR')
        plt.plot(new_sim['frequencies'][0] / 1e12, 10 * np.log10(new_sim['SNR_NL'][0]), 'g.', label='SNR_nl')
        plt.plot(new_sim['frequencies'][0] / 1e12, 10 * np.log10(new_sim['OSNR'][0]), 'r.', label='OSNR')
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


def plot_freq_gsnr(data_path, name, figure_path=None, save_fig=False, plot_fig=False, option=2):
    data = loadmat(data_path)
    key_gsnr, key_snr_li = None, None
    if option == 2:
        key_gsnr, key_snr_li = 'GSNR_2', 'SNR_NL_2'
        if key_gsnr not in data.keys() or key_snr_li not in data.keys():
            print('Invalid option. Using regular GSNR and SNR_nl')
            key_gsnr, key_snr_li = 'GSNR', 'SNR_NL'
    elif option == 1:
        key_gsnr, key_snr_li = 'GSNR', 'SNR_NL'
    else:
        print('Invalid option')
        exit()

    plt.figure(1)
    plt.plot(data['frequencies'][0] / 1e12, lin2db(data[key_gsnr][0]), 'b.', label='GSNR')
    plt.plot(data['frequencies'][0] / 1e12, lin2db(data[key_snr_li][0]), 'g.', label='SNR_nl')
    plt.plot(data['frequencies'][0] / 1e12, lin2db(data['OSNR'][0]), 'r.', label='OSNR')
    plt.title('C+L+S - {}'.format(name), fontsize=20)
    plt.ylabel('Power ratio (dB)', fontsize=14)
    plt.xlabel('Frequencies (THz)', fontsize=14)
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig((figure_path / '{}.png'.format(name)), bbox_inches='tight')
        plt.savefig((figure_path / '{}.pdf'.format(name)), bbox_inches='tight')

    if plot_fig:
        plt.show()


def plot_bp_traffic():
    pass


def plot_best_combinations(data_path, best_combinations, save_figs=False, plot_figs=True):
    figures_path = data_path / 'Figures'
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)

    for best in best_combinations:
        plot_freq_gsnr((data_path / (best_combinations[best] + '.mat')), best, figure_path=figures_path,
                       save_fig=save_figs, plot_fig=plot_figs, option=2)


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

    pareto = identify_pareto_max(population)
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


def plot_alloc_traffic(plot_figs=True, save_figs=False):
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Allocated_traffic/Allocated_traffic_Plot/DT_uniform')

    # for folder in list_folders:
    list_curves = []
    topology_traffic_path = path_data
    list_sub_folders = [file for file in os.listdir(path_data)
                        if os.path.isdir(path_data / file)]
    index_ref = [i for i, file in enumerate(list_sub_folders) if 'Reference' in file][0]
    ref_folder = list_sub_folders[index_ref]
    list_sub_folders.remove(ref_folder)
    list_sub_folders.insert(0, ref_folder)

    # Save all curves for a topology/Traffic
    for sub_folder in list_sub_folders:
        name = sub_folder
        mat_file = [file for file in os.listdir(path_data / sub_folder) if file.endswith('.mat')]
        mat_path = (path_data / (sub_folder + '/' + mat_file[0]))

        mat_data = loadmat(mat_path)
        average_accept_req = np.transpose(mat_data['cell_averageCumAcceptDemands'][0][0])
        norm_traffic_band = np.transpose(mat_data['cell_norm_traffic_band'][0][0])
        norm_traffic_lambda = np.transpose(mat_data['cell_norm_traffic_lambda'][0][0])
        prob_reject = np.transpose(mat_data['cell_probReject'][0][0])
        total_acc_traffic = np.transpose(mat_data['cell_totalAcceptedTraffic'][0][0])
        data = {'Name': name, 'Av_acc_req': average_accept_req, 'Norm_tra_band': norm_traffic_band,
                'Norm_tra_lambda': norm_traffic_lambda, 'Prob_reject': prob_reject,
                'Tot_acc_tra': total_acc_traffic}
        list_curves.append(data)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Configuration for allocated traffic
    plt.figure(figsize=(8, 4))
    for curve in list_curves:
        plt.plot(curve['Tot_acc_tra'], curve['Prob_reject'], label=curve['Name'], linewidth=2)
    plt.ylabel('Blocking probability', fontsize=18, fontweight='bold')
    plt.xlabel('Total Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.yscale('log')
    plt.ylim(1e-4, 2e-1)
    plt.xlim(300, 2100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    if save_figs:
        plt.savefig((topology_traffic_path / 'AllocTraffic_BP.png'), bbox_inches='tight', dpi=600)
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
        name = sub_folder
        mat_file = [file for file in os.listdir(path_data / sub_folder) if file.endswith('.mat')]
        mat_file = mat_file[0]
        mat_path = (path_data / (sub_folder + '/' + mat_file))

        mat_data = loadmat(mat_path)
        average_accept_req = np.transpose(mat_data['cell_averageCumAcceptDemands'][0][0])
        norm_traffic_band = np.transpose(mat_data['cell_norm_traffic_band'][0][0])
        norm_traffic_lambda = np.transpose(mat_data['cell_norm_traffic_lambda'][0][0])
        prob_reject = np.transpose(mat_data['cell_probReject'][0][0])
        total_acc_traffic = np.transpose(mat_data['cell_totalAcceptedTraffic'][0][0])
        data = {'Name': name, 'Av_acc_req': average_accept_req, 'Norm_tra_band': norm_traffic_band,
                'Norm_tra_lambda': norm_traffic_lambda, 'Prob_reject': prob_reject,
                'Tot_acc_tra': total_acc_traffic}
        list_curves.append(data)

    # Bold for all labels, legends...
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    # Configuration for allocated traffic
    plt.figure(figsize=(8, 4))
    for curve in list_curves:
        plt.plot(curve['Tot_acc_tra'], curve['Prob_reject'], label=curve['Name'], linewidth=2)
    plt.ylabel('Blocking probability', fontsize=18, fontweight='bold')
    plt.xlabel('Total Allocated traffic [Tbps]', fontsize=18, fontweight='bold')
    plt.yscale('log')
    plt.ylim(1e-4, 2e-1)
    plt.xlim(400, 1100)
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


def plot_gsnr():
    """
    Plot of GSNR for channels computed per band, loading from a .npy file
    """
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/Compare_GSNR')
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

    h, = plt.plot(freq_c / 1E12, gsnr_c, '.', marker=marker_c, c=color_c, markersize=marker_size,
                  label=(name + ' Band GSNR'))
    handles.append(copy.copy(h))
    plt.axvline(x=freq_c_begin / 1e12, color=color_c)
    plt.axvline(x=freq_c_end / 1e12, color=color_c)
    plt.hlines(y=np.mean(gsnr_c), xmin=freq_c_begin / 1e12, xmax=freq_c_end / 1e12, linestyles='dashdot',
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

    h, = plt.plot(freq_l / 1E12, gsnr_l_cl, '.', marker=marker_cl, c=color_l, markersize=marker_size,
                  label=(name + ' Band GSNR'))
    plt.plot(freq_c / 1E12, gsnr_c_cl, '.', marker=marker_cl, c=color_c, markersize=marker_size)
    handles.append(copy.copy(h))
    plt.axvline(x=freq_l_begin / 1e12, color=color_l)
    plt.axvline(x=freq_l_end / 1e12, color=color_l)
    plt.hlines(y=np.mean(gsnr_c_cl), xmin=freq_c_begin / 1e12, xmax=freq_c_end / 1e12, linestyles='dotted',
               colors=color_c)
    plt.hlines(y=np.mean(gsnr_l_cl), xmin=freq_l_begin / 1e12, xmax=freq_l_end / 1e12, linestyles='dotted',
               colors=color_l)

    # Plot for C+L+S
    name = 'C+L+S'
    data_cls = np.load(path_data / (name + '.npy'))
    freq_s = np.array([freq for freq in data_cls[0] if freq not in freq_cl])
    freq_s_begin = 196.55e12
    freq_s_end = 201.3e12
    gsnr_l_cls = [gsnr for i, gsnr in enumerate(data_cls[1]) if i < len(freq_c)]
    gsnr_c_cls = [gsnr for i, gsnr in enumerate(data_cls[1]) if len(freq_c) <= i < (2 * len(freq_c))]
    gsnr_s_cls = [gsnr for i, gsnr in enumerate(data_cls[1]) if i >= (2 * len(freq_c))]

    h, = plt.plot(freq_l / 1E12, gsnr_l_cls, '.', marker=marker_cls, c=color_l, markersize=marker_size,
                  label=(name + ' Band GSNR'))
    plt.plot(freq_c / 1E12, gsnr_c_cls, '.', marker=marker_cls, c=color_c, markersize=marker_size)
    plt.plot(freq_s / 1E12, gsnr_s_cls, '.', marker=marker_cls, c=color_s, markersize=marker_size)
    handles.append(copy.copy(h))
    plt.axvline(x=freq_s_begin / 1e12, color=color_s)
    plt.axvline(x=freq_s_end / 1e12, color=color_s)
    plt.hlines(y=np.mean(gsnr_c_cls), xmin=freq_c_begin / 1e12, xmax=freq_c_end / 1e12, linestyles='solid',
               colors=color_c)
    plt.hlines(y=np.mean(gsnr_l_cls), xmin=freq_l_begin / 1e12, xmax=freq_l_end / 1e12, linestyles='solid',
               colors=color_l)
    plt.hlines(y=np.mean(gsnr_s_cls), xmin=freq_s_begin / 1e12, xmax=freq_s_end / 1e12, linestyles='solid',
               colors=color_s)

    ax.text((np.mean(freq_c) / 1e12) - 1.2, 28, 'C-Band', fontsize=18, color=color_c,
            bbox=dict(facecolor='white', edgecolor=color_c))
    ax.text((np.mean(freq_l) / 1e12) - 1.2, 28, 'L-Band', fontsize=18, color=color_l,
            bbox=dict(facecolor='white', edgecolor=color_l))
    ax.text((np.mean(freq_s) / 1e12) - 1.2, 28, 'S-Band', fontsize=18, color=color_s,
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
    # plt.savefig((path_data / 'GSNR.png'), bbox_inches='tight', dpi=600)
    # plt.savefig((path_data / 'GSNR.pdf'), bbox_inches='tight', dpi=600)

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

    # Plots
    # plt.plot(freqs_l_band / 1e12, noise_figure_l_band, '.', marker=marker_c, markersize=8, color='red',
    #          label='Noise figure for L-band')
    # plt.plot(freqs_c_band / 1e12, noise_figure_c_band, '.', marker=marker_c, markersize=8, color='blue',
    #          label='Noise figure for C-band')
    plt.plot(freqs_s_band / 1e12, noise_figure_s_band, '.', marker=marker_c, markersize=8, color='green',
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


if __name__ == "__main__":
    plot_gsnr()
