from era5_compressor import ERA5Compressor, encode_wavelet_feedback
import pickle
import os
import netCDF4 as nc
import xarray as xr
from tqdm import tqdm
import numpy as np
from glob import glob
from argparse import ArgumentParser

class Dataset1Accessor:
    def __init__(self, path):
        self.ds = xr.open_dataset(path)

    def get_shape(self, variable):
        return self.ds[variable].shape

    def get_data(self, variable, time, level):
        return self.ds[variable].isel(time=time, plev=level).to_numpy()
    
class Dataset2Accessor:
    def __init__(self, path):
        self.ds = nc.MFDataset(path, aggdim="time")

    def get_shape(self, variable):
        shape = self.ds[variable].shape
        if len(shape) == 3:
            return shape[0], 1, shape[1], shape[2]
        else:
            return self.ds[variable].shape

    def get_data(self, variable, time, level):
        if len(self.ds[variable].shape) == 3:
            assert level == 0
            return np.array(self.ds[variable][time, :, :])
        else:
            return np.array(self.ds[variable][time, level, :, :])

def compress_dataset1(variable, month, CR_jp2=1000, CR_wavelet=1000, CR_wavelet_low=50, wavelet_level=2, copy_file=False):
    CR_overall = 1/(1/CR_jp2 + 1/CR_wavelet)
    variables = ["z", "q", "t", "u", "v"]
    assert variable in variables
    input_base_name = f"Z_1h_HRplev_alllevs_025_2016_{month:02}_{variable}.nc"
    output_base_name = f"Z_1h_HRplev_alllevs_025_2016_{month:02}"
    input_path = f"/scratch/lhuang/IAC/3D_alllevs_025/2016/{month:02}"
    output_path = f"/scratch/lhuang/IAC/3D_alllevs_025/2016/compressed/CR{int(CR_overall)}"
    decompressed_path = f"/users/lhuang/IAC/3D_alllevs_025/2016/decompressed/CR{int(CR_overall)}"
    min_path = f"/scratch/lhuang/IAC/3D_alllevs_025/2016/stats/{output_base_name}_min.nc"
    max_path = f"/scratch/lhuang/IAC/3D_alllevs_025/2016/stats/{output_base_name}_max.nc"

    os.system(f"mkdir -p {output_path}")
    os.system(f"mkdir -p {decompressed_path}")

    wavelet = "bior3.3"
    mode = "constant"

    #if exists file
    #if not os.path.exists(f"{decompressed_path}/{input_base_name}"):
        #os.system(f"rsync -P -ac {input_path}/{input_base_name} {decompressed_path}/{input_base_name}")
    #if not os.path.exists(f"{decompressed_path}/{input_base_name}"):
    if copy_file:
        os.system(f"rsync -P -ac {input_path}/{input_base_name} {decompressed_path}/{input_base_name}")
            


    one_minus_quantile = 1 / CR_wavelet
    ds_accessor = Dataset1Accessor(f"{input_path}/{input_base_name}")
    ds_decompressed = nc.Dataset(f"{decompressed_path}/{input_base_name}", "r+")
    
    print(f"Compressing variable: {variable}")
    compressor = ERA5Compressor(dataset_accessor=ds_accessor, 
                                output_path=output_path, output_base_name=output_base_name,
                                min_path=min_path, max_path=max_path, variable=variable, CR_jp2=CR_jp2)
    jp2_res = compressor.jp2_encoding()
    stat_res = compressor.compute_max_err_target(jp2_res, one_minus_quantile, nbins=1000)

    error_target = stat_res["error_target"]

    wavelet_file_size = 0
    nT, nP, nlat, nlon = ds_accessor.get_shape(variable)
    pbar = tqdm(total=nT * nP, desc=f"Pass 4 Sparse Wavelet")

    for it in range(nT):
        wavelet_coeffs = {}
        for ip in range(nP):
            jp2_pred = compressor.jp2_decoding(jp2_res["jp2_pred"], it, ip, nT, nP, jp2_res["level_min"], jp2_res["level_max"])
            data_ref = ds_accessor.get_data(variable, it, ip)
            delta = data_ref - jp2_pred
            coeffs_byte, delta_sp = encode_wavelet_feedback(wavelet, delta, 5000, error_target, cratio_low=CR_wavelet_low, level=wavelet_level, mode=mode)
            wavelet_coeffs[ip] = coeffs_byte
            ds_decompressed[variable][it, ip, :, :] = jp2_pred + delta_sp
            pbar.set_postfix(max_err=f"{np.max(np.abs(delta_sp - delta)):.2f}", target_err=f"{error_target:.2f}")
            pbar.update(1)
        wavelet_filename = f"{output_path}/{output_base_name}_{variable}_{it}.pkl"
        pickle.dump(wavelet_coeffs, open(wavelet_filename, "wb"))
        wavelet_file_size += os.path.getsize(wavelet_filename)

    pbar.close()


    pickle.dump(stat_res, open(f"{output_path}/{output_base_name}_{variable}_stat.pkl", "wb"))

    jp2_file_size = jp2_res["jp2_pred_size"]
    CR_actual = (nT * nP * nlat * nlon * 4) / (jp2_file_size + wavelet_file_size)
    print(f"Variable: {variable}, File size: {(jp2_file_size + wavelet_file_size) / (1024 * 1024)} MB, CR_actual: {CR_actual}")
    ds_decompressed.close()

# TODO: parallelize over days
# TODO: accumulate stat with different days & hours
def compress_dataset2(day, CR_jp2=1000, CR_wavelet=1000, CR_wavelet_low=50, wavelet_level=2, copy_file=False):
    CR_overall = 1/(1/CR_jp2 + 1/CR_wavelet)
    
    input_path = f"/scratch/lhuang/IAC/reference.0.25/original"
    output_path = f"/scratch/lhuang/IAC/reference.0.25/compressed/CR{int(CR_overall)}"
    decompressed_path = f"/users/lhuang/IAC/reference.0.25/decompressed/CR{int(CR_overall)}"
    min_path = f"/scratch/lhuang/IAC/reference.0.25/stats/P201609_min.nc"
    max_path = f"/scratch/lhuang/IAC/reference.0.25/stats/P201609_max.nc"
    variables = ["Q", "T", "OMEGA", "U", "V", "PS"]
    os.system(f"mkdir -p {output_path}")
    os.system(f"mkdir -p {decompressed_path}")

    wavelet = "bior3.3"
    mode = "constant"

    input_base_name = f"P201609{day:02}*"
    output_base_name = f"P201609{day:02}"

    if copy_file:
        # c for checksum, u for ignore newer files
        os.system(f"rsync -P -ac {input_path}/{input_base_name} {decompressed_path}/")

    one_minus_quantile = 1 / CR_wavelet

    jp2_dict = {}
    stat_dict = {}
    coords_dict = {}
    
    wavelet_file_size = 0
    jp2_file_size = 0


    ds_accessor = Dataset2Accessor(f"{input_path}/{input_base_name}")
    for variable in variables:
        print(f"Compressing variable: {variable}")
        compressor = ERA5Compressor(dataset_accessor=ds_accessor, 
                                    output_path=output_path, output_base_name=output_base_name,
                                    min_path=min_path, max_path=max_path, variable=variable, CR_jp2=CR_jp2)
        jp2_res = compressor.jp2_encoding()
        coords_dict[variable] = ds_accessor.get_shape(variable)
        stat_res = compressor.compute_max_err_target(jp2_res, one_minus_quantile, nbins=1000)
        pickle.dump(stat_res, open(f"{output_path}/{output_base_name}_{variable}_stat.pkl", "wb"))
        jp2_dict[(day, variable)] = jp2_res
        stat_dict[(day, variable)] = stat_res
        jp2_file_size += jp2_res["jp2_pred_size"]
        del jp2_res, stat_res

    nT, nP, nlat, nlon = coords_dict[variables[0]]
    pbar = tqdm(total=nT * nP * (len(variables) - 1) + nT * 1, desc=f"Pass 4 Sparse Wavelet")
    
    for hour in range(0, 24):
        it = hour
        ds_decompressed = nc.Dataset(f"{decompressed_path}/P201609{day:02}_{hour:02}", "r+")        
        for variable in variables:
            wavelet_coeffs = {}
            error_target = stat_dict[(day, variable)]["error_target"]
            jp2_res = jp2_dict[(day, variable)]
            nT, nP, _, _ = coords_dict[variable]
            slice_dim = len(ds_decompressed[variable].shape)
            for ip in range(nP):
                jp2_pred = compressor.jp2_decoding(jp2_res["jp2_pred"], it, ip, nT, nP, jp2_res["level_min"], jp2_res["level_max"])
                data_ref = ds_accessor.get_data(variable, it, ip)
                delta = data_ref - jp2_pred
                coeffs_byte, delta_sp = encode_wavelet_feedback(wavelet, delta, 5000, error_target, cratio_low=CR_wavelet_low, level=wavelet_level, mode=mode)
                wavelet_coeffs[ip] = coeffs_byte
                if slice_dim == 4:
                    ds_decompressed[variable][0, ip, :, :] = jp2_pred + delta_sp
                elif slice_dim == 3:
                    ds_decompressed[variable][0, :, :] = jp2_pred + delta_sp
                else:
                    raise NotImplementedError
                pbar.set_postfix(max_err=f"{np.max(np.abs(delta_sp - delta)):.2f}", target_err=f"{error_target:.2f}", hour=hour, variable=variable)
                pbar.update(1)
            wavelet_filename = f"{output_path}/P201609{day:02}_{hour:02}_{variable}.pkl"
            pickle.dump(wavelet_coeffs, open(wavelet_filename, "wb"))
            wavelet_file_size += os.path.getsize(wavelet_filename)
        ds_decompressed.close()

    nT, nP, nlat, nlon = coords_dict[variables[0]]
    CR_actual = (nT * nP * nlat * nlon * (len(variables) - 1) + nT * nlat * nlon )*4 / (jp2_file_size + wavelet_file_size)
    print(f"File size: {(jp2_file_size + wavelet_file_size) / (1024 * 1024)} MB, CR_actual: {CR_actual}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--CR_jp2", type=int, default=1000)
    parser.add_argument("--CR_wavelet", type=int, default=1000)
    parser.add_argument("--CR_wavelet_low", type=int, default=50)
    parser.add_argument("--variable", type=str, default="z")
    parser.add_argument("--wavelet_level", type=int, default=2)
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument("--day", type=int, default=1)
    parser.add_argument("--copy_file", action="store_true")
    args = parser.parse_args()

    if args.dataset == 1:
        compress_dataset1(variable=args.variable, month=args.month, CR_jp2=args.CR_jp2, CR_wavelet=args.CR_wavelet, CR_wavelet_low=args.CR_wavelet_low, wavelet_level=args.wavelet_level, copy_file=args.copy_file)
    elif args.dataset == 2:
        compress_dataset2(day=args.day, CR_jp2=args.CR_jp2, CR_wavelet=args.CR_wavelet, CR_wavelet_low=args.CR_wavelet_low, wavelet_level=args.wavelet_level, copy_file=args.copy_file)
    else:
        raise NotImplementedError

# sbatch -p long -c 32 -n 1 -N 1 -t 24:00:00 -A g34 --mem=80G -o compress_dataset1_%j.out -J comp_iac --wrap="python compress_iac.py"
