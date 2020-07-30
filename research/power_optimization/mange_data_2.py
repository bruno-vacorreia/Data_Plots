import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from gnpy.core.utils import lin2db
from research.power_optimization.plot_figures import plot_best_combinations, plot_freq_powers, plot_freq_gsnr
from research.power_optimization.data import DataTraffic, DataQot


def calc_best_gsnr(data_path, output_folder=None, thr=0.01, copy_files=False):
    """
    Save a list of profiles with best GSNR average for a threshold value of difference
    """
    config_path = data_path / 'config_file.json'
    if not output_folder:
        output_folder = data_path.parent / 'results/results_{}'.format(thr)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Create list of .mat files (combinations) and load configuration file
    list_data_files = [file for file in os.listdir(data_path) if file.endswith('.mat')]
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']

    # Calculation only for the channels under test
    ordered_list = []
    files_bar = tqdm(iterable=list_data_files, desc='Calculating gsnr average for all combinations')
    for file_name in files_bar:
        data = DataQot.load_qot_mat(data_path / file_name)
        calc_gsnr_lin = [gsnr for i, gsnr in enumerate(data.gsnr) if i in data.cut_index]
        calc_gsnr_db = [lin2db(gsnr) for gsnr in calc_gsnr_lin]
        gsnr_average = np.mean(calc_gsnr_db)
        ordered_list.append([data.name, gsnr_average])

    # Order list of combinations by GSNR average
    ordered_list.sort(key=lambda value: value[1], reverse=True)
    thr_gsnr = ordered_list[0][1] - (ordered_list[0][1] * thr)

    # Remove items with smaller GSNR average than the threshold
    ordered_list = [item for item in ordered_list if item[1] >= thr_gsnr]
    print('{} profiles with acceptable GSNR average'.format(len(ordered_list)))

    # Save best GSNR list and profiles
    with open((output_folder / 'Best_GSNR_average_{}.txt'.format(thr)), 'w') as best_combs:
        for comb in ordered_list:
            best_combs.write('{}: {}\n'.format(comb[0], comb[1]))
            if copy_files:
                shutil.copy2((data_path / (comb[0] + '.mat')), output_folder)


def calc_best_flatness(data_path, output_folder=None, thr=0.01, num_profiles=1, copy_files=False):
    """
    Save a list of best GSNR profiles regarding flatness. Read the name of the files from .txt, using the
    threshold value.
    """
    # Load configuration file
    config_path = data_path / 'config_file.json'
    config = json.load(open(config_path, 'r'))
    bands = config['spectral_config']['bands']

    # Output folder
    if not output_folder:
        output_folder = data_path.parent / 'results/results_{}'.format(thr)

    # Get best GSNR profiles from .txt file
    list_best = []
    with open((output_folder / 'Best_GSNR_average_{}.txt'.format(thr)), 'r') as file_best_comb:
        data = file_best_comb.readlines()
        for row in data:
            list_best.append(row.split(': ')[0])

    # Calculate the flatness
    ordered_list = []
    files_bar = tqdm(iterable=list_best, desc='Calculating best combinations flatness')
    for file_name in files_bar:
        data = DataQot.load_qot_mat(data_path / (file_name + '.mat'))
        calc_gsnr_lin = [gsnr for i, gsnr in enumerate(data.gsnr) if i in data.cut_index]
        calc_gsnr_db = [lin2db(gsnr) for gsnr in calc_gsnr_lin]
        gsnr_average = np.mean(calc_gsnr_db)

        # Calculate max, min and delta GSNR per band
        delta_gsnr_band = []
        num_ch_band = 0
        for band in bands:
            calc_gsnr_band = [gsnr for gsnr in calc_gsnr_db[num_ch_band:(num_ch_band +
                                                                         len(bands[band]['cut_indexes']))]]
            max_gsnr_band = max(calc_gsnr_band)
            min_gsnr_band = min(calc_gsnr_band)
            delta_gsnr_band.append(max_gsnr_band - min_gsnr_band)
            num_ch_band += len(bands[band]['comp_channels'])

        ordered_list.append([data.name, gsnr_average, np.mean(delta_gsnr_band)])

    # Order the list regarding flatness
    ordered_list.sort(key=lambda item: item[2])
    if num_profiles < len(ordered_list):
        names = [item[0] for item in ordered_list[0:num_profiles]]
        flatness = [item[2] for item in ordered_list[0:num_profiles]]
    else:
        names = [item[0] for item in ordered_list]
        flatness = [item[2] for item in ordered_list]

    # Save best GSNR list and profiles
    with open((output_folder / 'Best_flatness_{}.txt'.format(thr)), 'w') as best_flat:
        for i, name in enumerate(names):
            best_flat.write('{}: {}\n'.format(name, flatness[i]))
            if copy_files:
                shutil.copy2((data_path / (name + '.mat')), output_folder)


def plot_best_profiles(data_path, output_folder=None, thr=0.01):
    # Output folder
    if not output_folder:
        output_folder = data_path.parent / 'results/results_{}'.format(thr)

    # Get best GSNR profiles from .txt file
    list_best = []
    try:
        with open((output_folder / 'Best_flatness_{}.txt'.format(thr)), 'r') as file_best_comb:
            data = file_best_comb.readlines()
            for row in data:
                list_best.append(row.split(': ')[0])
    except FileNotFoundError:
        print('Threshold value not computed\nPath: {}'.format(output_folder))
        exit()

    output_folder = output_folder / 'Figures/GSNR'

    files_bar = tqdm(iterable=list_best, desc='Plotting best GSNR profiles')
    for file_name in files_bar:
        plot_freq_gsnr(data_path=(data_path / (file_name + '.mat')), figure_path=output_folder, save_fig=True)


def plot_power_profiles(data_path, output_folder=None, thr=0.01):
    # Output folder
    if not output_folder:
        output_folder = data_path.parent / 'results/results_{}'.format(thr)

    # Get best GSNR profiles from .txt file
    list_best = []
    try:
        with open((output_folder / 'Best_flatness_{}.txt'.format(thr)), 'r') as file_best_comb:
            data = file_best_comb.readlines()
            for row in data:
                list_best.append(row.split(': ')[0])
    except FileNotFoundError:
        print('Threshold value not computed\nPath: {}'.format(output_folder))
        exit()

    output_folder = output_folder / 'Figures/Powers'

    files_bar = tqdm(iterable=list_best, desc='Plotting powers of best GSNR profiles')
    for file_name in files_bar:
        plot_freq_powers(data_path=(data_path / (file_name + '.mat')), figure_path=output_folder, save_fig=True)


def plot_all_profiles(data_path, output_folder=None):
    """
    Plot all GSNR profiles in a folder.
    """
    # Output folder
    if not output_folder:
        output_folder = data_path.parent / 'Figures'

    list_data_files = [file for file in os.listdir(data_path) if file.endswith('.mat')]
    files_bar = tqdm(iterable=list_data_files, desc='Plotting gsnr profiles for all files')
    for file_name in files_bar:
        plot_freq_gsnr(data_path=(data_path / file_name), figure_path=output_folder, save_fig=True)


def calc_gsnr_average(data_path):
    """
    Print the GSNR average, maximum and minimum value of all .mat files in folder.
    """
    list_data_files = [file for file in os.listdir(data_path) if file.endswith('.mat')]

    for file in list_data_files:
        data = DataQot.load_qot_mat(path=(data_path / file))
        gsnr = [value for value in data.gsnr]
        gsnr_db = [lin2db(value) for value in gsnr]
        print('{}'.format(file))
        print('Average: {}'.format(np.average(gsnr_db)))
        print('Minimum: {}'.format(np.min(gsnr_db)))
        print('Maximum: {}'.format(np.max(gsnr_db)))


def calculate_alloc_traffic(data_path, bp_thr=1e-2):
    """
    Print the parameters of SNAP results for a BP threshold value
    """
    # List and organizes folders in main data folder
    list_curves = []
    list_sub_folders = [file for file in os.listdir(data_path) if os.path.isdir(data_path / file)]
    index_ref = [i for i, file in enumerate(list_sub_folders) if 'Reference' in file][0]
    ref_folder = list_sub_folders[index_ref]
    list_sub_folders.remove(ref_folder)
    list_sub_folders.insert(0, ref_folder)

    # Save all curves for a topology/Traffic
    for sub_folder in list_sub_folders:
        mat_file = [file for file in os.listdir(data_path / sub_folder) if file.endswith('.mat')]
        if mat_file:
            mat_path = (data_path / (sub_folder + '/' + mat_file[0]))
            data = DataTraffic.load_traffic_mat(path=mat_path, name=sub_folder)
            list_curves.append(data)

    for curve in list_curves:
        print('\nName: {}'.format(curve.name))
        for i, bp in enumerate(curve.prob_rejected):
            if bp >= bp_thr:
                print('Blokcing probability: {}'.format(curve.prob_rejected[i]))
                print('Average accepted request: {}'.format(curve.ave_acc_req[i]))
                print('Normalized traffic band: {}'.format(curve.norm_traffic_band[i]))
                print('Normalized traffic lambda: {}'.format(curve.norm_traffic_lambda[i]))
                print('Total allocated traffic: {}'.format(curve.total_acc_traffic[i]))
                break


if __name__ == "__main__":
    # path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
    #             'JOCN_Power_Optimization/C_L_S/Data_combinations/Test_profiles_flatNF_0.5/JOCN_2020')
    # path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
    #             'Ema_split_step/Data_CL_75WDM/Old_Att_C/Test')
    path = Path('/mnt/Bruno_Data/GoogleDrive/Material_Academico/PoliTo_WON/Research/Simulations_Data/Results/'
                'Power_comsumption/First_GSNR_profiles/01nm/C')

    # Allocated traffic
    # calculate_alloc_traffic(data_path=path, bp_thr=1e-1)
    # calc_gsnr_average(data_path=path)

    # Parameters / Best launch power
    # threshold = 0.002
    # number_profiles = 3
    # calc_best_gsnr(data_path=path, thr=threshold, copy_files=False)
    # calc_best_flatness(data_path=path, thr=threshold, num_profiles=number_profiles, copy_files=True)
    # plot_best_profiles(data_path=path, thr=threshold)
    # plot_power_profiles(data_path=path, thr=threshold)

    # General profiles
    # plot_all_profiles(data_path=path)
