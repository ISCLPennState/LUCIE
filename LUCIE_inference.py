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

data_dict = torch.load('LUCIE_training_data.pth')
input_dataset = data_dict['input_dataset']
target_dataset = data_dict['target_dataset']
input_means = data_dict['input_means']
input_stds = data_dict['input_stds']
input_mins = data_dict['input_mins']
input_maxs = data_dict['input_maxs']
target_means = data_dict['target_means']
target_stds = data_dict['target_stds']
target_mins = data_dict['target_mins']
target_maxs = data_dict['target_maxs']


ntrain = 14000
ntest = 2000

train_a = input_dataset[:ntrain,:,:,:]
train_u = target_dataset[:ntrain,:,:,:]


test_a = input_dataset[ntrain:ntrain+ntest,:,:,:]
test_u = target_dataset[ntrain:ntrain+ntest,:,:,:]


train_set = TensorDataset(train_a, train_u)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

test_set = TensorDataset(test_a, test_u)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)



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
                 num_layers=7, in_chans=7, out_chans=6, scale_factor=1, embed_dim=48, activation_function="silu", big_skip=True, pos_embed="latlon", use_mlp=True,
                                          normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,
                                          mlp_ratio = 2.).to(device)

state_pth = torch.load("LUCIE.pth")
model.load_state_dict(state_pth)

from tqdm import tqdm
val_a = input_dataset[0,:,:,:]

save_data = []
model.eval()
with torch.no_grad():
    inp_val = val_a.reshape(1,7,48,96)
    inp_val = inp_val.to(device)
    for i in tqdm(range(7500)):
        previous = inp_val[:,:5,:,:]

        pred = model(inp_val)
        pred[:,:5,:,:] = pred[:,:5,:,:] * target_stds[:,:5,:,:]

        pred[:,:5,:,:] += previous[:,:5,:,:] * input_stds + input_means
        tp_frame = pred[:,5:,:,:] * tp_std + tp_mean
        # pred_frame += (previous_frame + 1) / 2 * (input_maxs - input_mins) + input_mins
        plot = torch.cat((pred[:,:5,:,:],tp_frame), 1)

        inp_val = (pred[:,:5,:,:] - input_means) / input_stds
        inp_val = torch.cat((inp_val, input_dataset[i+1,5:,:,:].reshape(1,2,48,96).to(device)), dim=1)
        plot = plot.cpu().clone().detach().permute(0,2,3,1).numpy()
        save_data.append(plot[0])

save_data = np.array(save_data)
save_data[:,:,:,5] = (np.exp(save_data[:,:,:,5]) - 1) * 1e-2

np.savez('LUCIE.npz', single_array=save_data)


