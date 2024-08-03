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



def train_model(model, train_set, test_set, optimizer, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=8, loss_fn='l2', reg_rate=0):

    train_start = time.time()

    train_loader_1 = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader_1 = DataLoader(test_set, batch_size=16, shuffle=True)

    infer_bias = 1e+20
    recall_count = 0
    for epoch in range(nepochs):
        print(epoch)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        if epoch < 149:
            if scheduler is not None:
                scheduler.step()
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
        if recall_count > 10:
            print(epoch)
            break
        epoch_start = time.time()

        optimizer.zero_grad()
        acc_loss = 0

        model.train()
        batch_num = 0
        for inp, tar in train_loader_1:
            #noise = (torch.randn(inp.size(0), 5, inp.size(2), inp.size(3)) * std + mean).to(device)
            batch_num += 1
            loss = 0
            inp = inp.to(device)
            #inp[:,:5,:,:] += noise
            tar = tar.to(device)
            #tar[:,:5,:,:] -= noise
            prd = model(inp)

            if loss_fn == 'l2':
                loss_delta = l2loss_sphere(prd[:,:5,:,:], tar[:,:5,:,:], relative=True)
                loss_tp = torch.mean((prd[:,5:,:,:]-tar[:,5:,:,:])**2)
                loss = (loss_delta.to(device)*5 + loss_tp.to(device))/6

            elif loss_fn == "spectral l2":

                loss = SpectralLoss(img_size=(48, 96), absolute=True)(prd.to("cpu"),tar.to("cpu"))


            prd = prd.to(device)
            roll_loss = 0
            acc_loss += loss.item() * inp.size(0)
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prd_plt = prd
            tar_plt = tar


        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_frame = input_dataset[0].reshape(1,7,48,96) # T, SH, U, V, SP, TISR, ORO
                pred_frame = pred_frame.to(device)
                temp_bias = torch.zeros(48,96).to(device)
                for k in range(7500):
                    previous_frame = pred_frame[:,:5,:,:]
                    pred_frame = model(pred_frame) # 6
                    temp_bias += pred_frame[0,0,:,:].clone().detach()
                    pred_frame[:,:5,:,:] = pred_frame[:,:5,:,:] * target_stds[:,:5,:,:] # T, SH, U, V, SP, TP
                    # pred_frame = (pred_frame + 1) / 2 * (target_maxs - target_mins) + target_mins

                    pred_frame[:,:5,:,:] += previous_frame[:,:5,:,:] * input_stds + input_means
                    tp_frame = pred_frame[:,5:,:,:] * tp_std + tp_mean
                    # pred_frame += (previous_frame + 1) / 2 * (input_maxs - input_mins) + input_mins
                    plot_frame = torch.cat((pred_frame[:,:5,:,:], tp_frame), 1) # T, SH, U, V, SP, TP

                    pred_frame = pred_frame[:,:5,:,:]
                    pred_frame = (pred_frame - input_means) / input_stds
                    # pred_frame = 2 * (pred_frame - input_mins) / (input_maxs - input_mins) - 1

                    pred_frame = torch.cat((pred_frame, input_dataset[k+1,5:,:,:].reshape(1,2,48,96).to(device)), dim=1)
                    
                temp_bias = torch.mean(torch.abs(temp_bias / 7500 - true_temp_clim))
                print(temp_bias)
                if epoch > 100:
                    if temp_bias <= infer_bias:
                        infer_bias = temp_bias
                        torch.save(model.state_dict(), 'LUCIE_method_results/LUCIE_training_checkpoint.pth')
                        recall_count = 0
                    else:
                        state_pth = torch.load('LUCIE_method_results/LUCIE_training_checkpoint.pth')
                        model.load_state_dict(state_pth)
                        recall_count += 1
                        
                    
                plot_frame = plot_frame.clone().detach().cpu().numpy()
                fig = plt.figure(figsize=(14, 4))
                plt.subplot(1,7,1)
                plt.imshow(plot_frame[0][0], origin="lower")
                plt.axis("off")
                plt.colorbar(shrink=0.3)
                plt.subplot(1,7,2)
                plt.imshow(plot_frame[0][1], origin="lower")
                plt.axis("off")
                plt.colorbar(shrink=0.3)
                plt.subplot(1,7,3)
                plt.imshow(plot_frame[0][2], origin="lower")
                plt.axis("off")
                plt.colorbar(shrink=0.3)
                plt.subplot(1,7,4)
                plt.imshow(plot_frame[0][3], origin="lower")
                plt.axis("off")
                plt.colorbar(shrink=0.3)
                plt.subplot(1,7,5)
                plt.imshow(plot_frame[0][4], origin="lower")
                plt.axis("off")
                plt.colorbar(shrink=0.3)
                plt.subplot(1,7,6)
                plt.imshow(plot_frame[0][5], origin="lower")
                plt.axis("off")
                plt.colorbar(shrink=0.3)
                plt.show()





torch.manual_seed(1447)
torch.cuda.manual_seed(1447)
torch.cuda.empty_cache()

std = 0.05
mean = 0

grid='legendre-gauss'
nlat = 48
nlon = 96
hard_thresholding_fraction = 0.9
lmax = ceil(nlat / 1)
mmax = lmax
modes_lat = int(nlat * hard_thresholding_fraction)
modes_lon = int(nlon//2 * hard_thresholding_fraction)
modes_lat = modes_lon = min(modes_lat, modes_lon)
sht = RealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=grid, csphase=False)
radius=6.37122E6
cost, quad_weights = th.quadrature.legendre_gauss_weights(nlat, -1, 1)
#cost, quad_weights = th.quadrature.clenshaw_curtiss_weights(nlat, -1, 1)
quad_weights = (torch.as_tensor(quad_weights).reshape(-1, 1)).to(device)

model = SphericalFourierNeuralOperatorNet(params = {}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(48, 96),
                 num_layers=7, in_chans=7, out_chans=6, scale_factor=1, embed_dim=48, activation_function="silu", big_skip=True, pos_embed="latlon", use_mlp=True,
                                          normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,
                                          mlp_ratio = 2.).to(device)

from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
train_model(model, train_set, test_set, optimizer, scheduler=scheduler, nepochs=150, loss_fn = "l2")
# torch.save(model.state_dict(), 'LUCIE.pth')
