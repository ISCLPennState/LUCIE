from functools import partial
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Any, Tuple
import torch_harmonics as th
import torch_harmonics.distributed as thd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import time


from torch_harmonics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp
import math
from math import ceil, sqrt
import warnings

from torch.cuda import amp

# import FactorizedTensor from tensorly for tensorized operations
import tensorly as tl

tl.set_backend("pytorch")
# from tensorly.plugins import use_opt_einsum
# use_opt_einsum('optimal')
from tltorch.factorized_tensors.core import FactorizedTensor

import os
import logging
import datetime as dt
from typing import Union
import numpy as np
from typing import Tuple

from torch_harmonics.quadrature import legendre_gauss_weights, lobatto_weights, clenshaw_curtiss_weights
from torch_harmonics.legendre import _precompute_legpoly, _precompute_dlegpoly
from torch_harmonics.distributed import polar_group_size, azimuth_group_size, distributed_transpose_azimuth, distributed_transpose_polar
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim

from LUCIE_utils import *

from dataclasses import dataclass


from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Optional, Tuple

def geometric_mean(array):
    log_array = np.log(array)
    log_mean = np.mean(log_array)
    geom_mean = np.exp(log_mean)

    return geom_mean

def residual_norm(array, std_array):
    tar = array[1:]-array[:-1]
    std = std_zeromean(tar)
    geo_mean = geometric_mean(std_array)
    std_res = std / geo_mean
    return array / std_res, std_res

def std_zeromean(array):
    return np.sqrt(np.mean( array ** 2 ))

def load_data(fname):
    data = np.load(fname)
    data_list = []
    for file in data.files:
        data_list.append(data[file])

    np_data = np.asarray(data_list)
    np_data = np.transpose(np_data, (1, 2, 3, 0))

    return np_data

#data = load_data("../speedy_numpy_file.npz")
data = load_data('era5_numpy_file.npz')

data_sr = np.load("era5_tisr.npz")
tisr = data_sr["tisr"]
tisr = (tisr - np.mean(tisr))/np.std(tisr)

# data_sst = np.load("era5_sst.npz")
# sst_true = data_sst["sst"]
# sst_true[np.isnan(sst_true)] = 270
# sst = (sst_true - np.mean(sst_true))/np.std(sst_true)

std_array = []
temp = data[...,0]
temp_mean = np.mean(temp)
temp_std = np.std(temp)
temp = (temp - np.mean(temp))/np.std(temp)
#temp = 2*(temp - np.min(temp)) / (np.max(temp) - np.min(temp))-1

humid = data[...,1]
humid_mean = np.mean(humid)
humid_std = np.std(humid)
humid = (humid - np.mean(humid))/np.std(humid)
#humid = 2*(humid - np.min(humid)) / (np.max(humid) - np.min(humid))-1

u_wind = data[...,2]
u_wind_mean = np.mean(u_wind)
u_wind_std = np.std(u_wind)
u_wind = (u_wind - np.mean(u_wind))/np.std(u_wind)
#u_wind = 2*(u_wind - np.min(u_wind)) / (np.max(u_wind) - np.min(u_wind))-1

v_wind = data[...,3]
v_wind_mean = np.mean(v_wind)
v_wind_std = np.std(v_wind)
v_wind = (v_wind - np.mean(v_wind))/np.std(v_wind)

#stacked_data = np.stack((temp, humid, u_wind, v_wind), axis=-1)
#stacked_data = (stacked_data - np.mean(stacked_data))/np.std(stacked_data)

# temp = stacked_data[...,0]
# humid = stacked_data[...,1]
# u_wind = stacked_data[...,2]
# v_wind = stacked_data[...,3]

std_array.append(std_zeromean(temp[1:]-temp[:-1]))
std_array.append(std_zeromean(humid[1:]-humid[:-1]))
std_array.append(std_zeromean(u_wind[1:]-u_wind[:-1]))
std_array.append(std_zeromean(v_wind[1:]-v_wind[:-1]))
# std_array.append(std_zeromean(tisr[1:]-tisr[:-1]))
# std_array.append(std_zeromean(sst[1:]-sst[:-1]))
std_array = np.array(std_array)

_, temp_res = residual_norm(temp, std_array)
_, humid_res = residual_norm(humid, std_array)
_, u_res = residual_norm(u_wind, std_array)
_, v_res = residual_norm(v_wind, std_array)
# _, tisr_res = residual_norm(tisr, std_array)
#sst, sst_res = residual_norm(sst, std_array)

res_array = np.stack((temp_res, humid_res, u_res, v_res), axis=0).reshape(1,4,1,1)
res_array = torch.tensor(res_array).to(device)

stacked_data = np.stack((temp, humid, u_wind, v_wind, tisr), axis=-1)
#stacked_res = np.stack((temp_res, humid_res, u_res, v_res), axis=-1)


#stacked_data = np.stack((temp,humid), axis=-1)


ntrain = 14000
ntest = 2000


dataset = stacked_data
dataset = torch.tensor(dataset)



train_a = dataset[:ntrain,:,:,:]
train_u = dataset[1:ntrain+1,:,:,:]
train_a = train_a.permute(0,3,1,2)
train_u = train_u.permute(0,3,1,2)


test_a = dataset[ntrain:ntrain+ntest,:,:,:]
test_u = dataset[ntrain+1:ntrain+ntest+1,:,:,:]
test_a = test_a.permute(0,3,1,2)
test_u = test_u.permute(0,3,1,2)
dataset = dataset.permute(0,3,1,2)



torch.manual_seed(1447)
torch.cuda.manual_seed(1447)
torch.cuda.empty_cache()

grid='legendre-gauss'
nlat = 48
nlon = 96
hard_thresholding_fraction = 1
lmax = ceil(nlat / 1)
mmax = lmax
modes_lat = int(nlat * hard_thresholding_fraction)
modes_lon = int(nlon//2 * hard_thresholding_fraction)
modes_lat = modes_lon = min(modes_lat, modes_lon)
sht = DistributedRealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=grid, csphase=False)
radius=6.37122E6
cost, quad_weights = th.quadrature.legendre_gauss_weights(nlat, -1, 1)
#cost, quad_weights = th.quadrature.clenshaw_curtiss_weights(nlat, -1, 1)
quad_weights = (torch.as_tensor(quad_weights).reshape(-1, 1)).to(device)

model = SphericalFourierNeuralOperatorNet(params = {}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(48, 96),
                 num_layers=6, in_chans=5, out_chans=4, scale_factor=1, embed_dim=32, activation_function="gelu", big_skip=True, pos_embed="latlon", use_mlp=True,
                                          normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,
                                          mlp_ratio = 2.).to(device)

state_pth = torch.load("ACE_era5_tisr_lg_specl2_zscore_0.9_sp.pth")
model.load_state_dict(state_pth)

save_data = []
model.to(device)
model.eval()
with torch.no_grad():
    inp_val = dataset[0].reshape(1,6,48,96).to(device)
    for i in tqdm(range(7500)):
        pred = model(inp_val)
        #inp_val += data_denorm(pred.clone().detach(), val_mean, val_std)
        #inp_val = inp_val + pred
        inp_val = torch.cat((pred, dataset[i+1,5:,:,:].reshape(1,1,48,96).to(device)), dim=1)

        pred = pred.cpu().clone().detach().permute(0,2,3,1).numpy()
        save_data.append(pred[0])

save_data = np.array(save_data)
np.savez('ACE_era5_tisr_5years.npz', single_array=save_data)


