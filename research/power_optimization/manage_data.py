import pandas as pd
import numpy as np
import os
import json
import shutil
import collections
from scipy.constants import c
from tqdm import tqdm
from scipy.interpolate import interp1d
from pathlib import Path
from scipy.io import loadmat, savemat
from gnpy.core.utils import lin2db, db2lin
from research.power_optimization.plot_figures import plot_freq_gsnr, plot_best_combinations
from research.power_optimization.data import DataQot


def create_noise_figures_c_l_bands():
    or_path = '/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/' \
              'Research/Scripts/GSNR/infinera_nf_interpolated.mat'

    file_data = loadmat(or_path)

    noise_figure_db = file_data['NF_dB']
    noise_figure_db = [it.tolist()[0] for it in list(noise_figure_db)]
    noise_figure_average_db = file_data['NF_dB_avg']
    noise_figure_average_db = [it.tolist()[0] for it in list(noise_figure_average_db)]
    frequencies = file_data['f_channel']
    frequencies = [it.tolist()[0] for it in list(frequencies)]

    or_path = Path(__file__).parent / 'noise_figures'
    num_channels = len(frequencies)
    num_band = int(num_channels / 2)

    band = 'l'
    noise_figure_db_band = noise_figure_db[0:num_band]
    noise_figure_average_db_band = noise_figure_average_db[0:num_band]
    frequencies_band = frequencies[0:num_band]
    file_data = {'frequencies': frequencies_band, 'noise_figure': noise_figure_db_band}
    file_data = pd.DataFrame(file_data)
    path = or_path / (band + '_band_noise_figure.csv')
    file_data.to_csv(path_or_buf=(or_path / path), index=False)
    data_average = {'frequencies': frequencies_band, 'noise_figure_average': noise_figure_average_db_band}
    data_average = pd.DataFrame(data_average)
    path = or_path / (band + '_band_noise_figure_average.csv')
    data_average.to_csv(path_or_buf=(or_path / path), index=False)

    band = 'c'
    noise_figure_db_band = noise_figure_db[num_band:num_channels + 1]
    noise_figure_average_db_band = noise_figure_average_db[num_band:num_channels + 1]
    frequencies_band = frequencies[num_band:num_channels + 1]
    file_data = {'frequencies': frequencies_band, 'noise_figure': noise_figure_db_band}
    file_data = pd.DataFrame(file_data)
    path = or_path / (band + '_band_noise_figure.csv')
    file_data.to_csv(path_or_buf=(or_path / path), index=False)
    data_average = {'frequencies': frequencies_band, 'noise_figure_average': noise_figure_average_db_band}
    data_average = pd.DataFrame(data_average)
    path = or_path / (band + '_band_noise_figure_average.csv')
    data_average.to_csv(path_or_buf=(or_path / path), index=False)


def create_dispersion_file(data_path):
    sheet = 'Sheet1'
    data = pd.read_excel(io=data_path, sheet_name=sheet, header=None, names=['Wavelength', 'min', 'max'])
    wavelength_list = data.loc[(slice(None)), 'Wavelength'].to_numpy()
    min_list = data.loc[(slice(None)), 'min'].to_numpy()
    max_list = data.loc[(slice(None)), 'max'].to_numpy()
    dispersion_list = np.array([np.mean([min_list[i], max_list[i]]) for i in range(len(min_list))])
    dispersion_list = dispersion_list * 1e-6
    freq_list = (c / (wavelength_list * 1e-9))
    data['frequencies'] = freq_list
    data['dispersion'] = dispersion_list
    data.drop(labels=['Wavelength', 'min', 'max'], axis=1, inplace=True)
    data.to_csv(path_or_buf='/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/OpticalSystemInterface/'
                            'resources/power_optimization/input_data/dispersion/G.652.D.csv', index=False)


def create_cut_indices(data_path):
    config_path = data_path / 'config_file.json'
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']

    num_ch_band = 0
    for band in bands.values():
        band['cut_indexes'] = [(ch + num_ch_band - 1) for ch in band['comp_channels']]
        num_ch_band += band['nb_channels']

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def recalc_all_gsnr(res_path):
    config_path = res_path / 'config_file.json'

    list_data_files = [file for file in os.listdir(res_path) if file.endswith('.mat')]
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']

    files_bar = tqdm(iterable=list_data_files, desc='Recalculating all gsnr files')
    for file in files_bar:
        data = DataQot.load_qot_mat(res_path / file)
        recalc_gsnr(data, bands)
        data.save_data()


def recalc_gsnr(data, bands_dict):
    # Apply the interpolation in each band, using only the computed channels in each band
    cut_index, init_freq_index = 0, 0
    new_gsnr, new_nli = np.array([]), np.array([])
    for band in bands_dict.values():
        band_freq = [value for i, value in enumerate(data.frequencies) if
                     i in range(init_freq_index, init_freq_index + band['nb_channels'])]
        init_freq_index += band['nb_channels']
        cut_freq = [value for i, value in enumerate(data.frequencies) if i in band['cut_indexes']]
        band_nli = [nli for i, nli in enumerate(data.snr_nl) if i in band['cut_indexes']]
        func_nli = interp1d(cut_freq, band_nli, kind='linear', fill_value='extrapolate')
        band_nli = func_nli(band_freq)
        new_nli = np.append(new_nli, band_nli, 0)
    new_gsnr = (1 / ((1 / new_nli) + (1 / data.osnr)))

    # Add the news values for SNR_nl and GSNR
    data.snr_nl = new_nli
    data.gsnr = new_gsnr


def find_best_combination():
    """Function to find the best GSNR profile in average and flatness"""
    root_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Data_combinations/Data_preprocesses_Sband_96_bestCh/JOCN_2020')
    config_path = root_path / 'config_file.json'
    best_comb_folder = root_path.parent / 'results/Best_combinations'
    if not os.path.isdir(best_comb_folder):
        os.makedirs(best_comb_folder)

    list_data_files = [file for file in os.listdir(root_path) if file.endswith('.mat')]
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']
    best_combinations = {}
    # Indices of computed channels per band
    num_ch_band = 0
    for band in bands:
        bands[band]['comp_channels_ind'] = [(ch + num_ch_band - 1) for ch in bands[band]['comp_channels']]
        num_ch_band += bands[band]['nb_channels']

    # Initial values for gsnr and flatness
    best_gsnr = 0.0
    best_delta_gsnr = np.inf
    best_delta_gsnr_band = np.inf
    best_largest_delta_gsnr_band = np.inf

    # Calculation only for the channels under test
    files_bar = tqdm(iterable=list_data_files, desc='Calculating best gsnr files')
    for file_name in files_bar:
        data = loadmat(root_path / file_name)
        comb_name = str(data['name'][0])
        channels_indices = data['cut_index'][0]
        calc_gsnr_lin = [gsnr for i, gsnr in enumerate(data['GSNR'][0]) if i in channels_indices]
        calc_gsnr_db = [lin2db(gsnr) for gsnr in calc_gsnr_lin]

        # Largest average GSNR
        average_gsnr = np.mean(calc_gsnr_db)
        if average_gsnr > best_gsnr:
            best_combinations['best_gsnr_average'] = comb_name
            best_gsnr = average_gsnr

        # Smallest delta GSNR
        delta_gsnr = max(calc_gsnr_db) - min(calc_gsnr_db)
        if delta_gsnr < best_delta_gsnr:
            best_delta_gsnr = delta_gsnr
            best_combinations['best_delta_gsnr'] = comb_name

        # Calculate max, min and delta GSNR per band
        delta_gsnr_band = []
        num_ch_band = 0
        for band in bands:
            calc_gsnr_band = [gsnr for gsnr in calc_gsnr_db[num_ch_band:(num_ch_band +
                                                                         len(bands[band]['comp_channels']))]]
            max_gsnr_band = max(calc_gsnr_band)
            min_gsnr_band = min(calc_gsnr_band)
            delta_gsnr_band.append(max_gsnr_band - min_gsnr_band)
            num_ch_band += len(bands[band]['comp_channels'])

        # Smallest average delta GSNR (per band)
        if np.mean(delta_gsnr_band) < best_delta_gsnr_band:
            best_delta_gsnr_band = np.mean(delta_gsnr_band)
            best_combinations['best_delta_gsnr_band'] = comb_name

        if max(delta_gsnr_band) < best_largest_delta_gsnr_band:
            best_largest_delta_gsnr_band = max(delta_gsnr_band)
            best_combinations['best_largest_delta_gsnr_band'] = comb_name

    # Save and print best options found
    with open((best_comb_folder / 'Best_Combinations.txt'), 'w') as file_best_comb:
        for best in best_combinations:
            shutil.copy2((root_path / (best_combinations[best] + '.mat')), best_comb_folder)
            file_best_comb.write('{}: {}\n'.format(best, best_combinations[best]))
            print('{}: {}'.format(str(best), best_combinations[best]))

    plot_best_combinations(best_comb_folder, best_combinations, save_figs=True, plot_figs=False)


def find_best_gsnr():
    """Function to find the best GSNR profile in average and flatness"""
    root_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Data_combinations/Data_processes_Sband_96_bestCh/JOCN_2020')
    config_path = root_path / 'config_file.json'
    best_comb_folder = root_path.parent / 'results/Best_flat_among_best_gsnr'
    if not os.path.isdir(best_comb_folder):
        os.makedirs(best_comb_folder)

    threshold = 0.01

    list_data_files = [file for file in os.listdir(root_path) if file.endswith('.mat')]
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']
    best_combinations = {}
    # Indices of computed channels per band
    num_ch_band = 0
    for band in bands:
        bands[band]['comp_channels_ind'] = [(ch + num_ch_band - 1) for ch in bands[band]['comp_channels']]
        num_ch_band += bands[band]['nb_channels']

    # Calculation only for the channels under test
    ordered_list = []
    files_bar = tqdm(iterable=list_data_files, desc='Calculating gsnr average files')
    for file_name in files_bar:
        data = loadmat(root_path / file_name)
        comb_name = str(data['name'][0])
        channels_indices = data['cut_index'][0]
        calc_gsnr_lin = [gsnr for i, gsnr in enumerate(data['GSNR'][0]) if i in channels_indices]
        calc_gsnr_db = [lin2db(gsnr) for gsnr in calc_gsnr_lin]
        gsnr_average = np.mean(calc_gsnr_db)
        ordered_list.append([comb_name, gsnr_average])

    ordered_list.sort(key=lambda value: value[1], reverse=True)
    difference = ordered_list[0][1] - (ordered_list[0][1] * threshold)
    ordered_list = [item for item in ordered_list if item[1] > difference]
    print('{} profiles with good GSNR average'.format(len(ordered_list)))
    with open((best_comb_folder / 'Best_Combinations.txt'), 'w') as best_combs:
        for comb in ordered_list:
            best_combs.write('{}: {}\n'.format(comb[0], comb[1]))

    best_delta_gsnr_band = np.inf
    files_bar = tqdm(iterable=ordered_list, desc='Calculating best flatness among best gsnr files')
    for item in files_bar:
        item_name = item[0]
        item_gsnr = item[1]
        data = loadmat(root_path / (item_name + '.mat'))
        channels_indices = data['cut_index'][0]
        calc_gsnr_lin = [gsnr for i, gsnr in enumerate(data['GSNR'][0]) if i in channels_indices]
        calc_gsnr_db = [lin2db(gsnr) for gsnr in calc_gsnr_lin]

        # Calculate max, min and delta GSNR per band
        delta_gsnr_band = []
        num_ch_band = 0
        for band in bands:
            calc_gsnr_band = [gsnr for gsnr in calc_gsnr_db[num_ch_band:(num_ch_band +
                                                                         len(bands[band]['comp_channels']))]]
            max_gsnr_band = max(calc_gsnr_band)
            min_gsnr_band = min(calc_gsnr_band)
            delta_gsnr_band.append(max_gsnr_band - min_gsnr_band)
            num_ch_band += len(bands[band]['comp_channels'])

        # Smallest average delta GSNR (per band)
        if np.mean(delta_gsnr_band) < best_delta_gsnr_band:
            best_delta_gsnr_band = np.mean(delta_gsnr_band)
            best_combinations['best_flatness_best_gsnr'] = item_name

    # Save and print best options found
    with open((best_comb_folder / 'Best_Combination.txt'), 'w') as file_best_comb:
        for best in best_combinations:
            shutil.copy2((root_path / (best_combinations[best] + '.mat')), best_comb_folder)
            file_best_comb.write('{}: {}\n'.format(best, best_combinations[best]))
            print('{}: {}'.format(str(best), best_combinations[best]))

    plot_best_combinations(best_comb_folder, best_combinations, save_figs=True, plot_figs=False)


def check_best_gsnr():
    """Function to find the best GSNR profile in average and flatness"""
    root_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Data_combinations/Data_processes_Sband_96_bestCh/JOCN_2020')
    config_path = root_path / 'config_file.json'
    best_comb_folder = root_path.parent / 'results/Best_flat_among_best_gsnr'
    if not os.path.isdir(best_comb_folder):
        os.makedirs(best_comb_folder)
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']
    best_combinations = {}
    # Indices of computed channels per band
    num_ch_band = 0
    for band in bands:
        bands[band]['comp_channels_ind'] = [(ch + num_ch_band - 1) for ch in bands[band]['comp_channels']]
        num_ch_band += bands[band]['nb_channels']

    list_best = []
    with open((best_comb_folder / 'Best_Combinations.txt'), 'r') as file_best_comb:
        data = file_best_comb.readlines()
        for row in data:
            list_best.append(row.split(': ')[0])

    list_best = list_best[0:10]
    plot_best_combinations(root_path, list_best[0:10], save_figs=True, plot_figs=False)
    threshold = 0.01


def calc_average_gsnr():
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/Compare_GSNR')
    config_path = path_data / 'config_file_cl.json'
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']

    data = loadmat(path_data / 'jocn_2020_l_c_offset_l=0.5_tilt_l=0.1_offset_c=1.0_tilt_c=-0.3.mat')

    num_ch_band = 0
    for band in bands:
        bands[band]['comp_channels_ind'] = [(ch + num_ch_band - 1) for ch in bands[band]['comp_channels']]
        num_ch_band += bands[band]['nb_channels']

    gsnr_per_band = 0.0
    total_gsnr, freq_list = [], []
    for band in bands:
        gsnr_per_band = [lin2db(gsnr) for i, gsnr in enumerate(data['GSNR'][0])
                         if i in bands[band]['comp_channels_ind']]
        total_gsnr.extend(gsnr_per_band)
        freq_list.extend([freq for i, freq in enumerate(data['frequencies'][0])
                          if i in bands[band]['comp_channels_ind']])
        print('Average GSNR for {} band is {}'.format(band, np.mean(gsnr_per_band)))

    print('Average GSNR is {}'.format(np.mean(total_gsnr)))
    array = np.array([freq_list, total_gsnr])
    np.save(path_data / 'C+L.npy', array)


def calc_average_gsnr_opt():
    """
    Function to print the average GSNR for all channels for several .mat data files
    """
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Future_scenarios_analyze/Profiles')

    list_folder = [file for file in os.listdir(path_data)]

    for folder in list_folder:
        print(folder)
        list_files = [file for file in os.listdir(path_data / folder)]
        for file in list_files:
            data = loadmat(path_data / (folder + '/' + file))
            gsnr = np.transpose(data['GSNR'])

            print('Name of profile: {}'.format(data['name'][0]))
            print('Average GSNR: {} dB'.format(np.mean(lin2db(gsnr))))
        print('\n')


def get_noise_figures_average():
    """
    Function to print the average noise figure (in the entire band) for C, L and S bands
    """
    noise_fig_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/OpticalSystemInterface/'
                          'resources/power_optimization/input_data/noise_figures')
    bands = ['l', 'c', 's']

    for band in bands:
        band_noise_fig_path = noise_fig_path / (band + '_band_noise_figure.csv')
        data = pd.read_csv(filepath_or_buffer=band_noise_fig_path)

        print('Average noise figure for {} band: {}'.format(band, np.mean(data.noise_figure)))


def compare_raman():
    default_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                        'JOCN_Power_Optimization/C_L_S/Data_combinations')
    new_path = default_path / 'Data_new_raman_solver/JOCN_2020'
    old_path = default_path / 'Data_processed/JOCN_2020'
    figures_path = default_path / 'Figures'
    if not os.path.isdir(figures_path):
        os.makedirs(figures_path)

    # Error threshold
    error_thr = 0.03

    list_files_new = [name for name in os.listdir(new_path) if name.endswith('.mat')]
    list_files_old = [name for name in os.listdir(old_path) if name.endswith('.mat')]

    # Check if the number of files is the same
    assert collections.Counter(list_files_new) == collections.Counter(list_files_old), \
        'Different files in folder to compare'

    # Check if there is mismatch between the files to compare
    list_unmatched_files = []
    for file in list_files_new:
        if file not in list_files_old:
            list_unmatched_files.append(file)
    assert list_unmatched_files == []

    # Compare all files of the two folders and print the ones that presents a difference larger then an error threshold
    files_bar = tqdm(iterable=list_files_new, desc='Plot all combinations')
    for file in files_bar:
        data_new = loadmat(new_path / file)
        data_old = loadmat(old_path / file)

        delta_snr_nl = np.mean(abs(data_new['SNR_NL'][0] - data_old['SNR_NL'][0]) / data_old['SNR_NL'][0])
        delta_gsnr = np.mean(abs(data_new['GSNR'][0] - data_old['GSNR'][0]) / data_old['GSNR'][0])
        if delta_snr_nl > error_thr:
            print('Error larger than threshold for snr nli in {}'.format(file))
        if delta_gsnr > error_thr:
            print('Error larger than threshold for gsnr in {}'.format(file))


def function():
    data_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Flat_Power_Profiles/Regular_S_band_96')
    data_path_new = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                         'JOCN_Power_Optimization/C_L_S/Flat_Power_Profiles/Regular_S_band_96_new')

    mat_files = [item for item in os.listdir(data_path) if item.endswith('.mat')]
    config_files = [item for item in os.listdir(data_path) if item.endswith('.json')]

    for file in mat_files:
        file_old = data_path / file
        file_new = data_path_new / file
        file = file.replace('.mat', '').replace('jocn_2020_', '').replace('_', ' ')

        plot_freq_gsnr(data_path=file_old, name=file, figure_path=(data_path / 'Figures'),
                       save_fig=True, plot_fig=False)
        plot_freq_gsnr(data_path=file_new, name=file, figure_path=(data_path_new / 'Figures'),
                       save_fig=True, plot_fig=False)


def expand_s_band(data_path):
    # Config file
    config_path = data_path / 'config_file.json'
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']
    si = config['spectral_config']['general_si']

    # Indices for S band
    old_index_s_band = 0
    for band in bands:
        if band == 's':
            old_index_s_band = [i for i in range(old_index_s_band, old_index_s_band + bands[band]['nb_channels'])]
        else:
            old_index_s_band += bands[band]['nb_channels']
    cl_index = [i for i in range(0, 192)]

    # Expansion of S band
    bands['s']['nb_channels'] = 96*2
    total_num_ch = 0
    for band in bands:
        total_num_ch += bands[band]['nb_channels']

    list_data_files = [file for file in os.listdir(data_path) if file.endswith('.mat')]
    files_bar = tqdm(iterable=list_data_files, desc='Expanding all combination files')
    for file in files_bar:
        data = loadmat(data_path / file)

        # Frequency
        old_freq = np.array([data['frequencies'][0][ind] for ind in old_index_s_band])
        cl_freq = np.array([data['frequencies'][0][ind] for ind in cl_index])
        new_freq = np.array([])
        for index in range(bands['s']['nb_channels']):
            if index < len(old_freq):
                new_freq = np.append(new_freq, old_freq[index])
            else:
                new_freq = np.append(new_freq, new_freq[-1] + si['wdm_grid'])

        # ASE, SNR_nl and GSNR
        gsnr = np.array([gsnr for i, gsnr in enumerate(data['GSNR'][0]) if i in old_index_s_band])
        cl_gsnr = np.array([gsnr for i, gsnr in enumerate(data['GSNR'][0]) if i in cl_index])
        snr_nl = np.array([gsnr for i, gsnr in enumerate(data['SNR_NL'][0]) if i in old_index_s_band])
        cl_snr_nl = np.array([gsnr for i, gsnr in enumerate(data['SNR_NL'][0]) if i in cl_index])
        osnr = np.array([gsnr for i, gsnr in enumerate(data['OSNR'][0]) if i in old_index_s_band])
        cl_osnr = np.array([gsnr for i, gsnr in enumerate(data['OSNR'][0]) if i in cl_index])

        func = interp1d(old_freq, gsnr, kind='linear', fill_value='extrapolate')
        gsnr = func(new_freq)
        func = interp1d(old_freq, snr_nl, kind='linear', fill_value='extrapolate')
        snr_nl = func(new_freq)
        func = interp1d(old_freq, osnr, kind='linear', fill_value='extrapolate')
        osnr = func(new_freq)

        data['frequencies'] = np.append(cl_freq, new_freq)
        data['SNR_NL'] = np.append(cl_snr_nl, snr_nl)
        data['OSNR'] = np.append(cl_osnr, osnr)
        data['GSNR'] = (((data['OSNR']) ** (-1)) + ((data['SNR_NL']) ** (-1))) ** (-1)

        savemat(data_path / file, data)


def gsnr_01(data_path):
    config_path = data_path / 'config_file.json'
    config = json.load(open(config_path, 'r'))
    list_data_files = [file for file in os.listdir(data_path) if file.endswith('.mat')]

    baud_rate = config['spectral_config']['general_si']['symbol_rates']
    ratio_01nm = lin2db(12.5e9 / baud_rate)

    for file in list_data_files:
        data = DataQot.load_qot_mat(data_path / file)

        gsnr_01_list = []
        for i, gsnr_lin in enumerate(data.gsnr):
            gsnr = lin2db(gsnr_lin) - ratio_01nm
            gsnr_01_list.append(db2lin(gsnr))

        data.gsnr = np.array(gsnr_01_list)
        data.save_data()


if __name__ == "__main__":
    path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                'Ema_split_step/Ema_Profiles/C')

    # recalc_all_gsnr(path)
    # create_cut_indices(path)
    # gsnr_01(path)
