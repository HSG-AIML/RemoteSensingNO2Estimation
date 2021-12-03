import os

os.environ["OMP_NUM_THREADS"] = "6" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys
import copy
from tqdm  import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import NO2PredictionDataset
from transforms import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model import get_model
from utils import load_data, load_data_light, set_seed, step
from train import eval_metrics, split_samples, train, test
import random

import mlflow

# parameters
samples_file = "data/samples_S2S5P_whole_timespan.csv"
# datadir = "/netscratch/lscheibenreif/eea"
datadir = "/ds2/remote_sensing/eea/whole-timespan"
verbose = True
sources = samples_file.split("_")[1]
frequency = "whole_timespan" if "whole_timespan" in samples_file else samples_file.split("_")[2].replace(".csv", "")
epochs = 30
batch_size = 50
runs = 10
result_dir = "/netscratch/lscheibenreif/tmp"
checkpoint = "checkpoints/pretrained_resnet50_LUC.model" # None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
early_stopping = False
lr = 0.001

checkpoint_name = "pretrained" if checkpoint is not None else "from_scratch"

experiment = "_".join([sources, checkpoint_name, frequency])
if verbose: print("Init. mlflow experiment:", experiment)

# mlflow.create_experiment(experiment)

if verbose:
    print(samples_file)
    print(datadir)
    print(sources)
    print(frequency)
    print(checkpoint)
    print(device)
    print("Loading samples...")

samples, stations = load_data_light(datadir, samples_file, frequency, sources)

loss = nn.MSELoss()
datastats = DatasetStatistics()
tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), Randomize(), ToTensor()])

performances_test = []
performances_val = []
performances_train = []

for run in tqdm(range(1, runs+1), unit="run"):

    with mlflow.start_run():
        mlflow.log_param("samples_file", samples_file)
        mlflow.log_param("datadir", datadir)
        mlflow.log_param("sources", sources)
        mlflow.log_param("frequency", frequency)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("result_dir", result_dir)
        mlflow.log_param("pretrained_checkpoint", checkpoint)
        mlflow.log_param("device", device)
        mlflow.log_param("early_stopping", early_stopping)
        mlflow.log_param("lr", lr)
        mlflow.log_param("run", run)

        # set the seed for this run
        set_seed(run)

        # initialize dataloaders + model
        if verbose: print("Initializing dataset")

        samples_train, samples_val, samples_test = split_samples(samples, list(stations.keys()), 0.2, 0.2)

        dataset_test = NO2PredictionDataset(datadir, samples_test, frequency, sources, transforms=tf, station_imgs=stations)
        dataset_train = NO2PredictionDataset(datadir, samples_train, frequency, sources, transforms=tf, station_imgs=stations)
        dataset_val = NO2PredictionDataset(datadir, samples_val, frequency, sources, transforms=tf, station_imgs=stations)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=False)
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
        dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)
        dataloader_train_for_testing = DataLoader(dataset_train, batch_size=1, num_workers=1, shuffle=False, pin_memory=False)

        if verbose: print("Initializing model")
        model = get_model(sources, device, checkpoint)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=1e6, min_lr=1e-7)

        if verbose: print("Start training")
        # train the model
        for epoch in range(epochs):
            model.train()
            loss_history = []

            for idx, sample in enumerate(dataloader_train):
                model_input = sample["img"].float().to(device)
                if "S5P" in sources:
                    s5p = sample["s5p"].float().unsqueeze(dim=1).to(device)
                    model_input = {"img" : model_input, "s5p" : s5p}
                y = sample["no2"].float().to(device)

                loss_epoch = step(model_input, y, model, loss, optimizer)

            scheduler.step(epoch)
            torch.cuda.empty_cache()
            loss_history.append(loss_epoch/idx)

            val_y, val_y_hat = test(sources, model, dataloader_val, device, datastats)

            valid_val = (val_y_hat < 100) & (val_y_hat > 0)
            eval_val = eval_metrics(val_y, val_y_hat)

            if early_stopping:
                # stop training if evaluation performance does not increase
                if epoch > 25 and sum(valid_val) > len(valid_val) - 5:
                    if eval_val[0] > np.mean([performances_val[-3][2], performances_val[-2][2], performances_val[-1][2]]):
                        # performance on evaluation set is decreasing
                        if verbose: print("Stop at epoch:", epoch)
                        break

            print("epoch:", epoch, eval_val)
            print("valid:", len(valid_val), sum(valid_val))
            performances_val.append([run, epoch] + eval_val)

        mlflow.log_param("epochs", epoch)

        test_y, test_y_hat = test(sources, model, dataloader_test, device, datastats)
        train_y, train_y_hat = test(sources, model, dataloader_train_for_testing, device, datastats)

        valid = (test_y_hat < 100) & (test_y_hat > 0)
        valid_train = (train_y_hat < 100) & (train_y_hat > 0)

        eval_test = eval_metrics(test_y, test_y_hat)
        eval_train = eval_metrics(train_y, train_y_hat)

        # save img of predictions as artifact
        img, (ax1,ax2) = plt.subplots(1,2, figsize=(12,7))
        for ax in (ax1,ax2):
            ax.set_xlim((0,100))
            ax.set_ylim((0,100))
            ax.plot((0,0),(100,100), c="red")
        ax1.scatter(test_y, test_y_hat)
        ax1.set_title("test")
        ax2.scatter(train_y, train_y_hat)
        ax2.set_title("train")
        mlflow.log_figure(img, "predictions.png")

        mlflow.log_metric("r2", eval_test[0])
        mlflow.log_metric("mae", eval_test[1])
        mlflow.log_metric("mse", eval_test[2])

        performances_test.append(eval_test)
        performances_train.append(eval_train)

    performances_val = pd.DataFrame(performances_val, columns=["run", "epoch", "r2", "mae", "mse"])
    performances_test = pd.DataFrame(performances_test, columns=["r2", "mae", "mse"])
    performances_train = pd.DataFrame(performances_train, columns=["r2", "mae", "mse"])


if checkpoint is not None: checkpoint_name = checkpoint.split("/")[1].split(".")[0]

# save results
if verbose: print("writing results...")
performances_test.to_csv(os.path.join(result_dir, "_".join([sources, str(checkpoint_name), frequency, "test", str(epochs), "epochs"]) + ".csv"), index=False)
performances_train.to_csv(os.path.join(result_dir, "_".join([sources, str(checkpoint_name), frequency, "train", str(epochs), "epochs"]) + ".csv"), index=False)
performances_val.to_csv(os.path.join(result_dir, "_".join([sources, str(checkpoint_name), frequency, "val", str(epochs), "epochs"]) + ".csv"), index=False)

# save the model
if verbose: print("writing model...")
torch.save(model.state_dict(), os.path.join(result_dir, "_".join([sources, str(checkpoint_name), frequency, str(epochs), "epochs"]) + ".model"))
if verbose: print("done.")

