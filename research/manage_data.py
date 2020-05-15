import pandas as pd
import numpy as np
import os
import json
import shutil
from scipy.interpolate import interp1d
from pathlib import Path
from scipy.io import loadmat, savemat
from gnpy.core.utils import lin2db
from research.plot_figures import plot_best_combinations


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


def recalc_all_gsnr():
    data_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Future_scenarios_analyze/3_band')
    config_path = data_path / 'config_file.json'

    list_data_files = [file for file in os.listdir(data_path) if file.endswith('.mat')]
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']

    num_ch_band = 0
    for band in bands:
        bands[band]['comp_channels_ind'] = [(ch + num_ch_band - 1) for ch in bands[band]['comp_channels']]
        num_ch_band += bands[band]['nb_channels']

    for file in list_data_files:
        data = loadmat(data_path / file)
        data = recalc_gsnr(data, bands, replace=True)
        savemat(data_path / file, data)


def recalc_gsnr(data, bands_dict, replace=False):
    """
    Function to correct the snr_nl and gsnr due the wrong interpolation in solver module. If replace option is enabled,
    change the values of SNRnl and GSNR.  If not, add two new parameters
    """
    channels_indices = data['cut_index'][0]
    # Get the SNR_nl and OSNR and calculate the GSNR for the computed channels
    calc_snr_nli_indices = np.array([snr for i, snr in enumerate(data['SNR_NL'][0]) if i in channels_indices])
    calc_osnr_indices = np.array([osnr for i, osnr in enumerate(data['OSNR'][0]) if i in channels_indices])
    calc_gsnr_indices = 1 / ((1 / calc_snr_nli_indices) + (1 / calc_osnr_indices))
    calc_gsnr_indices = list(calc_gsnr_indices)
    calc_snr_nli_indices = list(calc_snr_nli_indices)

    # Aplly the interpolation in each band, using only the computed channels in each band
    cut_index, freq_index = 0, 0
    new_gsnr, new_nli = [], []
    for band in bands_dict:
        cut_freq = [freq for i, freq in enumerate(data['frequencies'][0]) if i in bands_dict[band]['comp_channels_ind']]
        freq = [data['frequencies'][0][ind] for ind in range(freq_index, freq_index+bands_dict[band]['nb_channels'])]
        band_gsnr = [gsnr for i, gsnr in enumerate(calc_gsnr_indices) if i in range(cut_index, cut_index+len(cut_freq))]
        band_nli = [nli for i, nli in enumerate(calc_snr_nli_indices) if i in range(cut_index, cut_index+len(cut_freq))]
        func_gsnr = interp1d(cut_freq, band_gsnr, kind='linear', fill_value='extrapolate')
        func_nli = interp1d(cut_freq, band_nli, kind='linear', fill_value='extrapolate')
        new_gsnr.extend(func_gsnr(freq))
        new_nli.extend(func_nli(freq))
        cut_index += len(cut_freq)
        freq_index += bands_dict[band]['nb_channels']

    # Add the news values for SNR_nl and GSNR
    if not replace:
        data['SNR_NL_2'] = np.array(new_nli)
        data['GSNR_2'] = np.array(new_gsnr)
    else:
        data['SNR_NL'] = np.array(new_nli)
        data['GSNR'] = np.array(new_gsnr)

    return data


def find_best_combination():
    """Function to find the best GSNR profile in average and flatness"""
    root_path = Path(__file__).parent / 'results/JOCN_2020'
    config_path = root_path / 'config_file.json'
    best_comb_folder = Path(__file__).parent / 'results/Best_combinations'
    if not os.path.isdir(best_comb_folder):
        os.mkdir(best_comb_folder)

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
    for i, file_name in enumerate(list_data_files):
        print('Computing combination {}'.format(i))
        data = loadmat(root_path / file_name)
        if 'GSNR_2' not in data.keys() or 'SNR_NL_2' not in data.keys():
            data = recalc_gsnr(data, bands)
            savemat(root_path / file_name, data)
        comb_name = str(data['name'][0])
        channels_indices = data['cut_index'][0]     # Computed indices
        calc_gsnr_lin = [gsnr for i, gsnr in enumerate(data['GSNR'][0]) if i in channels_indices]
        # calc_gsnr_lin = [gsnr for i, gsnr in enumerate(data['GSNR_2'][0])]
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
            # calc_gsnr_band = [gsnr for gsnr in calc_gsnr_db[num_ch_band:(num_ch_band + bands[band]['nb_channels'])]]
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


def copy_files_in_folders():
    """Function to copy all .mat files from several folders to one single folder"""
    path_data = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/research/Simulations_Data/Results/'
                     'JOCN_Power_Optimization/C_L_S/Data/Data_Annapurna/')
    dest_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/research/OpticalSystemInterface/'
                     'resources/power_optimization/results/JOCN_2020')

    list_data_folders = [folder for folder in os.listdir(path_data) if os.path.isdir(path_data / folder)]
    config_file = [file for file in os.listdir(path_data) if file.endswith('.json')]

    print('Copying files...')
    for folder in list_data_folders:
        list_data = os.listdir(path_data / folder)

        for data in list_data:
            shutil.copy2((path_data / (folder + '/' + data)), dest_path)

        print('Finish folder {}'.format(folder))

    for file in config_file:
        shutil.copy2(path_data / file, dest_path)


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


def get_noise_figures_average():
    noise_fig_path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/OpticalSystemInterface/'
                          'resources/power_optimization/input_data/noise_figures')
    bands = ['l', 'c', 's']

    for band in bands:
        band_noise_fig_path = noise_fig_path / (band + '_band_noise_figure.csv')
        data = pd.read_csv(filepath_or_buffer=band_noise_fig_path)

        print('Average noise figure for {} band: {}'.format(band, np.mean(data.noise_figure)))


if __name__ == "__main__":
    get_noise_figures_average()
