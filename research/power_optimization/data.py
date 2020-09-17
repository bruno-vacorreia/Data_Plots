import numpy as np
from scipy.io import loadmat, savemat


class DataTraffic:

    def __init__(self, name, path, ave_acc_req, norm_traffic_band, norm_traffic_lambda, prob_rej, total_acc_traffic,
                 bp_thr=1e-2):
        self._name = name
        self._path = path
        self._average_accept_req = ave_acc_req
        self._norm_traffic_band = norm_traffic_band
        self._norm_traffic_lambda = norm_traffic_lambda
        self._prob_rejected = prob_rej
        self._total_accept_traffic = total_acc_traffic
        self._thr_bp = bp_thr
        self._allocated_traffic = None
        self._multi_factor = None

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self._path

    @property
    def ave_acc_req(self):
        return self._average_accept_req

    @property
    def norm_traffic_band(self):
        return self._norm_traffic_band

    @property
    def norm_traffic_lambda(self):
        return self._norm_traffic_lambda

    @property
    def prob_rejected(self):
        return self._prob_rejected

    @property
    def total_acc_traffic(self):
        return self._total_accept_traffic

    @property
    def thr_bp(self):
        return self._thr_bp

    @thr_bp.setter
    def thr_bp(self, value):
        self._thr_bp = value

    @property
    def alloc_traffic(self):
        return self._allocated_traffic

    @property
    def multi_factor(self):
        return self._multi_factor

    def calc_alloc_traffic(self):
        for i, bp in enumerate(self.prob_rejected):
            if bp >= self.thr_bp:
                self._allocated_traffic = self.total_acc_traffic[i]
                break

    def calc_multi_factor(self, base_traffic):
        self._multi_factor = round(self.alloc_traffic / base_traffic, 2)

    @staticmethod
    def load_traffic_mat(path, name=None, thr_bp=1e-2):
        mat_data = loadmat(path)
        average_accept_req, norm_traffic_band, norm_traffic_lambda, prob_reject, total_acc_traffic = [], [], [], [], []
        try:
            average_accept_req = np.array([value[0] for value in
                                           np.transpose(mat_data['cell_averageCumAcceptDemands'][0][0])])
            norm_traffic_band = np.array([value[0] for value in np.transpose(mat_data['cell_norm_traffic_band'][0][0])])
            norm_traffic_lambda = np.array([value[0] for value in
                                            np.transpose(mat_data['cell_norm_traffic_lambda'][0][0])])
            prob_reject = np.array([value[0] for value in np.transpose(mat_data['cell_probReject'][0][0])])
            total_acc_traffic = np.array([value[0] for value in
                                          np.transpose(mat_data['cell_totalAcceptedTraffic'][0][0])])
        except KeyError as e:
            print('Missing "{}" argument in .mat file'.format(e.args[0]))
            exit()

        data = DataTraffic(name, path, average_accept_req, norm_traffic_band, norm_traffic_lambda, prob_reject,
                           total_acc_traffic, thr_bp)
        data.calc_alloc_traffic()

        return data


class DataQot:

    def __init__(self, path, name, num_bands, frequencies, powers, snr_nl, osnr, gsnr, ase, gain, cut_freq, cut_index):
        self._path = path
        self._name = name
        self._num_bands = num_bands
        self._frequencies = frequencies
        self._powers = powers
        self._snr_nl = snr_nl
        self._osnr = osnr
        self._gsnr = gsnr
        self._ase = ase
        self._gain = gain
        self._cut_freq = cut_freq
        self._cut_index = cut_index

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def num_bands(self):
        return self._num_bands
    
    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.setter
    def frequencies(self, value: np.ndarray):
        self._frequencies = value

    @property
    def powers(self):
        return self._powers

    @property
    def snr_nl(self):
        return self._snr_nl

    @snr_nl.setter
    def snr_nl(self, value: np.ndarray):
        self._snr_nl = value

    @property
    def osnr(self):
        return self._osnr

    @osnr.setter
    def osnr(self, value: np.ndarray):
        self._osnr = value
    
    @property
    def gsnr(self):
        return self._gsnr
    
    @gsnr.setter
    def gsnr(self, value: np.ndarray):
        self._gsnr = value

    @property
    def ase(self):
        return self._ase

    @property
    def gain(self):
        return self._gain

    @property
    def cut_fre(self):
        return self._cut_freq

    @property
    def cut_index(self):
        return self._cut_index

    def save_data(self):
        mat_data = loadmat(self._path)

        try:
            mat_data['name'] = self._name
            mat_data['num_bands'] = self._num_bands
            mat_data['frequencies'] = self._frequencies
            mat_data['powers'] = self._powers
            mat_data['SNR_NL'] = self._snr_nl
            mat_data['OSNR'] = self._osnr
            mat_data['GSNR'] = self._gsnr
            mat_data['ase'] = self._ase
            mat_data['G'] = self._gain
            mat_data['cut_freq'] = self._cut_freq
            mat_data['cut_index'] = self._cut_index
        except KeyError as e:
            print('Missing {} key in data'.format(e.args[0]))
            exit()

        savemat(self._path, mat_data)

    @staticmethod
    def load_qot_mat(path):
        mat_data = loadmat(path)
        data = None
        try:
            data = DataQot(path=path, name=mat_data['name'][0], num_bands=mat_data['num_bands'][0][0],
                           frequencies=mat_data['frequencies'][0], powers=mat_data['powers'][0],
                           snr_nl=mat_data['SNR_NL'][0], osnr=mat_data['OSNR'][0], gsnr=mat_data['GSNR'][0],
                           ase=mat_data['ase'][0], gain=mat_data['G'][0], cut_freq=mat_data['cut_freq'][0],
                           cut_index=mat_data['cut_index'][0])
        except KeyError as e:
            print('Missing {} argument in data file'.format(e.args[0]))
            exit()

        return data
