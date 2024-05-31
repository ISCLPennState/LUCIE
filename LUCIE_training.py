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



# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

train_set = TensorDataset(train_a, train_u)
train_loader = DataLoader(train_set, batch_size=1, shuffle=False)


test_set = TensorDataset(test_a, test_u)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)



def integrate_grid(ugrid, dimensionless=False, polar_opt=0):

    dlon = 2 * torch.pi / nlon
    radius = 1 if dimensionless else radius
    if polar_opt > 0:
        out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
    else:
        out = torch.sum(ugrid * quad_weights * dlon * radius**2, dim=(-2, -1))
    return out

def l2loss_sphere(prd, tar, relative=False, squared=True):
    loss = integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def spectral_l2loss_sphere(prd, tar, relative=False, squared=True):
    # compute coefficients
    prd = prd.cpu()
    tar = tar.cpu()
    diff = (prd-tar)

    shtdiff = sht(diff)
    coeffs = torch.view_as_real(shtdiff)
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def train_model(model, train_set, test_set, optimizer, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=8, loss_fn='l2'):

    train_start = time.time()

    train_loader_1 = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader_1 = DataLoader(test_set, batch_size=16, shuffle=True)

    for epoch in range(nepochs):
        epoch_start = time.time()

        optimizer.zero_grad()
        acc_loss = 0

        model.train()
        i = 0
        for inp, tar in train_loader_1:
            i += 1
            loss = 0
            inp = inp.to(device)
            tar = tar[:,:4,:,:].to(device)
            prd = model(inp)

            if loss_fn == 'l2':
                loss = l2loss_sphere(prd, tar)
            #    loss += 0.1*l2loss_sphere(frame, next_frame)
            elif loss_fn == "spectral l2":
                #loss = spectral_l2loss_sphere(prd, tar)
                # prd = prd / res_array
                # tar = tar / res_array
                loss = SpectralLoss(img_size=(48, 96), absolute=True)(prd.cpu(),tar.cpu())
            prd = prd.to(device)
            roll_loss = 0
            acc_loss += loss.item() * inp.size(0)
            loss.backward()
            optimizer.step()
            prd_plt = prd
            tar_plt = tar



        if scheduler is not None:
            scheduler.step()
        acc_loss = acc_loss / len(train_loader_1.dataset)
        if epoch % 5 == 0:
            print(str(epoch))
            plt.subplot(1,2,1)
            plt.imshow(prd_plt[0,0,:,:].clone().detach().cpu(),origin="lower")
            plt.colorbar(shrink=0.5)
            plt.subplot(1,2,2)
            plt.imshow(tar_plt[0,0,:,:].clone().detach().cpu(),origin="lower")
            plt.colorbar(shrink=0.5)
            #plt.savefig("train_inc.png")
            plt.show()
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for inp, tar in test_loader_1:
                inp = inp.to(device)
                tar = tar[:,:4,:,:].to(device)
                prd = model(inp)
                if loss_fn == 'l2':
                    loss = l2loss_sphere(prd, tar, relative=True)
                elif loss_fn == "spectral l2":
                    # prd = prd / res_array
                    # tar = tar / res_array
                    loss = spectral_l2loss_sphere(prd, tar, relative=True)
                    #loss = SpectralLoss(img_size=(48, 96),p=5.0)(prd.cpu(),tar.cpu())
                prd = prd.to(device)

                valid_loss += loss.item() * inp.size(0)
                prd_testplt = prd
                tar_testplt = tar
        valid_loss = valid_loss / len(test_loader_1.dataset)
        epoch_time = time.time() - epoch_start






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

from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
optimizer = torch.optim.Adam(model.parameters(), lr=1E-4, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-6)
train_model(model, train_set, test_set, optimizer, scheduler=scheduler, nepochs=100, loss_fn = "spectral l2")
torch.save(model.state_dict(), 'ACE_era5_tisr_lg_specl2.pth')
