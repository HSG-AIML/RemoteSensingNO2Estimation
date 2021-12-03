import os
from re import S

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset

import torch
import random

import xarray as xr
import rioxarray

def step(x, y, model, loss, optimizer):
    y_hat = model(x).squeeze()
    loss_epoch = loss(y, y_hat)
    optimizer.zero_grad()
    loss_epoch.backward()
    optimizer.step()

    return loss_epoch.detach().cpu()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data_to_memory(sample, datadir, frequency, sources, s5p_dates):
    if sample.get("img") is None:
        sample["img"] = np.load(os.path.join(datadir, "sentinel-2", sample["img_path"]))

    if sample.get("s5p") is None and "S5P" in sources:
        s5p_data = xr.open_dataset(os.path.join(datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
        if frequency == "whole_timespan":
            sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
        else:
            datestr = sample["date_str"]
            time_idx = np.where(s5p_dates==datestr)[0].item()
            sample["s5p"] = s5p_data.isel(time=time_idx).tropospheric_NO2_column_number_density.values.squeeze()

        s5p_data.close()
    return sample

def load_data(datadir, samples_file, frequency, sources):
    """load samples to memory, returns array of samples and array of stations
    each sample is a dict"""
    assert(sources in ["S2", "S2S5P"])
    assert(frequency in ["whole_timespan", "monthly", "quarterly"])

    samples_df = pd.read_csv(samples_file, index_col="idx")
    samples_df = samples_df[np.isnan(samples_df.no2) == False]

    s5p_dates = None
    if frequency != "whole_timespan":
        sample = samples_df.iloc[0]
        s5p_sample = xr.open_dataset(os.path.join(datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
        if frequency == "quarterly":
            s5p_dates = np.array(["Q-" + str(dt.quarter) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
        elif frequency  == "monthly":
            s5p_dates = np.array([str(dt.month) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
        s5p_sample.close()

    samples = []
    stations = []
    for idx in tqdm(range(samples_df.shape[0])):
        sample = samples_df.iloc[idx].to_dict()
        sample["idx"] = idx
        sample = load_data_to_memory(sample, datadir, frequency, sources, s5p_dates)
        samples.append(sample)
        stations.append(sample["AirQualityStation"])

    return samples, stations

def load_s5p_to_memory(sample, datadir, frequency, sources, s5p_dates):
    if sample.get("s5p") is None and "S5P" in sources:
        s5p_data = xr.open_dataset(os.path.join(datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
        if frequency == "whole_timespan":
            sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
        else:
            datestr = sample["date_str"]
            time_idx = np.where(s5p_dates==datestr)[0].item()
            sample["s5p"] = s5p_data.isel(time=time_idx).tropospheric_NO2_column_number_density.values.squeeze()

        s5p_data.close()
    return sample

def load_data_light(datadir, samples_file, frequency, sources, n=None):
    """load samples to memory, returns array of samples and array of stations
    each sample is a dict"""
    assert(sources in ["S2", "S2S5P"])
    assert(frequency in ["whole_timespan", "monthly", "quarterly"])

    if not isinstance(samples_file, pd.DataFrame):
        samples_df = pd.read_csv(samples_file, index_col="idx")
    else:
        samples_df = samples_file
    samples_df = samples_df[np.isnan(samples_df.no2) == False]
    #samples_df = samples_df.iloc[0:100]
    #print(samples_df.shape)

    s5p_dates = None
    if frequency != "whole_timespan":
        sample = samples_df.iloc[0]
        s5p_sample = xr.open_dataset(os.path.join(datadir, "sentinel-5p", sample["s5p_path"])).rio.write_crs(4326)
        if frequency == "quarterly":
            s5p_dates = np.array(["Q-" + str(dt.quarter) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
        elif frequency  == "monthly":
            s5p_dates = np.array([str(dt.month) + "-" + str(dt.year) for dt in pd.to_datetime(s5p_sample.time.values)])
        s5p_sample.close()

    samples = []
    stations = {}
    for idx in tqdm(range(samples_df.shape[0])):
        sample = samples_df.iloc[idx].to_dict()
        sample["idx"] = idx
        sample = load_s5p_to_memory(sample, datadir, frequency, sources, s5p_dates)
        samples.append(sample)
        stations[sample["AirQualityStation"]] = np.load(os.path.join(datadir, "sentinel-2", sample["img_path"]))

        if n is not None:
            # optionally break dataloading early (for quick debugging)
            if idx == n:
                break

    return samples, stations
