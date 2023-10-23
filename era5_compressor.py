import os
import pickle
import time
import warnings

import glymur
import zstd
import pywt

import xarray as xr
import numpy as np
import pandas as pd

from tqdm import tqdm

glymur.set_option('lib.num_threads', 32)
warnings.filterwarnings("ignore")

def encode_wavelet_feedback(wavelet, delta_origin, cratio, max_error_target, cratio_low=50, level=3, mode="periodic"):
    max_error = np.abs(delta_origin.mean() - delta_origin).max()  # todo: why do we subtract the mean?
    cratio = cratio * 4
    if max_error <= max_error_target:
        delta_sp = np.zeros_like(delta_origin) + delta_origin.mean()
        coeffs_sp_list = pywt.wavedec2(delta_sp, wavelet, level=level, mode=mode)
        res = pywt.waverec2(coeffs_sp_list, wavelet, mode=mode)
        delta_sp = res[:delta_origin.shape[0], :delta_origin.shape[1]]
    else:
        coeffs_list = pywt.wavedec2(delta_origin, wavelet, level=level, mode=mode)
        coeffs_sp_list = []
        for coeff in coeffs_list:
            if isinstance(coeff, tuple):
                LH, HL, HH = coeff
                q_LH = np.quantile(np.abs(LH), 1 - 1 / cratio)
                q_HL = np.quantile(np.abs(HL), 1 - 1 / cratio)
                q_HH = np.quantile(np.abs(HH), 1 - 1 / cratio)
                LH_sp = np.where(np.abs(LH) > q_LH, LH, 0)
                HL_sp = np.where(np.abs(HL) > q_HL, HL, 0)
                HH_sp = np.where(np.abs(HH) > q_HH, HH, 0)
                coeff = (LH_sp, HL_sp, HH_sp)
            else:
                LL = coeff
                LL_ratio = delta_origin.size / LL.size
                if LL_ratio * 10 < cratio:
                    q_LL = np.quantile(np.abs(coeff), 1 - 1 / cratio * LL_ratio)
                    LL_sp = np.where(np.abs(coeff) > q_LL, coeff, LL.mean())
                    coeff = LL_sp
            coeffs_sp_list.append(coeff)
        res = pywt.waverec2(coeffs_sp_list, wavelet, mode=mode)
        delta_sp = res[:delta_origin.shape[0], :delta_origin.shape[1]]
        max_error = np.max(np.abs(delta_sp - delta_origin))
    if max_error > max_error_target and cratio > cratio_low * 4:
        return encode_wavelet_feedback(wavelet, delta_origin, cratio / 4 / 2, max_error_target, cratio_low=cratio_low, level=level, mode=mode)
    else:
        complevel = 22  # why?
        coeffs_byte = zstd.compress(pickle.dumps(coeffs_sp_list), complevel)
        return coeffs_byte, delta_sp


class ERA5Compressor:

    def __init__(self, dataset_accessor, 
                 output_path, output_base_name,
                 min_path, max_path,
                 variable, CR_jp2
                 #vdim="lev", londim="lon", latdim="lat",
                 ):

        self.dataset_accessor = dataset_accessor
        self.output_path = output_path
        self.variable = variable
        self.CR_jp2 = CR_jp2
        self.model_type = "jp2"

        self.ds_ref_max_path = max_path
        self.ds_ref_min_path = min_path

        self.jp2_filename = f"{self.output_path}/{output_base_name}_{variable}.jp2" 

    def jp2_encoding(self):

        level_max = xr.open_dataset(self.ds_ref_max_path)[self.variable].to_numpy().squeeze()
        level_min = xr.open_dataset(self.ds_ref_min_path)[self.variable].to_numpy().squeeze()

        if len(level_max.shape) == 0:
            level_max = level_max.reshape(1)
            level_min = level_min.reshape(1)

        nT, nP, nlat, nlon = self.dataset_accessor.get_shape(self.variable)
        assert len(level_max) == len(level_min) == nP

        maxval = np.iinfo(np.uint16).max

        jp2_start_time = time.time()

        jp2 = glymur.Jp2k(self.jp2_filename, irreversible=True,
                            shape=((nT * nlat, nP * nlon)), tilesize=(nlat, nlon),
                            cratios=[self.CR_jp2 // 2])

        for i, tw in tqdm(enumerate(jp2.get_tilewriters()), total=nT * nP, desc="Pass 1 JP2"):
            it = i // nP
            ip = i % nP
            data_ref = self.dataset_accessor.get_data(self.variable, it, ip)
            data_ref = (data_ref - level_min[ip]) / (level_max[ip] - level_min[ip])
            data_ref = (data_ref * maxval).astype(np.uint16)
            tw[:] = data_ref

        jp2_duration = time.time() - jp2_start_time

        file_size = os.path.getsize(self.jp2_filename)
        real_cratio = (nT * nP * nlat * nlon * 4) / file_size
        print(f"File size: {file_size / (1024 * 1024)} MB, jp2_encoding cratio: {real_cratio}")
        res = {"jp2_pred": jp2, "level_max": level_max, "level_min": level_min,
                "jp2_pred_size": file_size, "jp2_duration": jp2_duration}

        return res
    
    def jp2_decoding(self, jp2, it, ip, nT, nP, level_min, level_max):
        maxval = np.iinfo(np.uint16).max
        data_pred = jp2.read(tile=it * nP + ip).astype(np.float32) / maxval
        data_pred = data_pred * (level_max[ip] - level_min[ip]) + level_min[ip]
        return data_pred


    def compute_max_err_target(self, jp2_encoding_res, one_minus_quantile, nbins=1000):

        comp_max_err_target_start_time = time.time()

        maxval = np.iinfo(np.uint16).max

        nT, nP, nlat, nlon = self.dataset_accessor.get_shape(self.variable)
        jp2_pred = jp2_encoding_res["jp2_pred"]
        level_max = jp2_encoding_res["level_max"]
        level_min = jp2_encoding_res["level_min"]

        gmin = np.inf
        gmax = -np.inf

        for i in tqdm(range(nT * nP), desc="Pass 2 Calc min&max"):
            it = i // nP
            ip = i % nP

            data_ref = self.dataset_accessor.get_data(self.variable, it, ip)
            data_pred = self.jp2_decoding(jp2_pred, it, ip, nT, nP, level_min, level_max)
            delta = np.abs(data_ref - data_pred)

            gmin = min(gmin, delta.min())
            gmax = max(gmax, delta.max())

        # generate histogram
        bins = np.linspace(gmin, gmax, nbins)
        hist = np.zeros_like(bins)[:-1]

        for i in tqdm(range(nT * nP), desc="Pass 3 Calc hist"):
            it = i // nP
            ip = i % nP

            data_ref = self.dataset_accessor.get_data(self.variable, it, ip)
            data_pred = jp2_pred.read(tile=it * nP + ip).astype(np.float32) / maxval
            data_pred = data_pred * (level_max[ip] - level_min[ip]) + level_min[ip]
            delta = np.abs(data_ref - data_pred)
            hist += np.histogram(delta, bins=bins, density=False)[0]

        hist = hist / hist.sum()

        cdf_rev = np.cumsum(hist[::-1])
        ind = np.searchsorted(cdf_rev, one_minus_quantile)
        error_target = bins[-ind-1-1]

        comp_max_err_target_duration = time.time() - comp_max_err_target_start_time

        res = {"bins": bins, "hist": hist, "error_target": error_target, "comp_max_err_target_duration": comp_max_err_target_duration}

        return res
