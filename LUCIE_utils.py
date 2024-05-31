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


from torch_harmonics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp
import math
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


@dataclass
class ModelMetaData:
    """Data class for storing essential meta data needed for all Modulus Models"""

    # Model info
    name: str = "ModulusModule"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = False
    amp_cpu: bool = None
    amp_gpu: bool = None
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    onnx_gpu: bool = None
    onnx_cpu: bool = None
    onnx_runtime: bool = False
    trt: bool = False
    # Physics informed
    var_dim: int = -1
    func_torch: bool = False
    auto_grad: bool = False

    def __post_init__(self):
        self.amp_cpu = self.amp if self.amp_cpu is None else self.amp_cpu
        self.amp_gpu = self.amp if self.amp_gpu is None else self.amp_gpu
        self.onnx_cpu = self.onnx if self.onnx_cpu is None else self.onnx_cpu
        self.onnx_gpu = self.onnx if self.onnx_gpu is None else self.onnx_gpu

import torch
import logging

from typing import Union
from pathlib import Path


class Module(torch.nn.Module):
    """The base class for all network models in Modulus.

    This should be used as a direct replacement for torch.nn.module

    Parameters
    ----------
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None
    """

    def __init__(self, meta: ModelMetaData = None):
        super().__init__()

        if not meta or not isinstance(meta, ModelMetaData):
            self.meta = ModelMetaData()
        else:
            self.meta = meta

        self.logger = logging.getLogger("core.module")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

        # dummy buffer for getting where the networks device
        self.register_buffer("device_buffer", torch.empty(0))

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # TODO: set up debug log
        # fh = logging.FileHandler(f'modulus-core-{self.meta.name}.log')

    def save(self, file_name: Union[str, None] = None) -> None:
        """Simple utility for saving just the model

        Parameters
        ----------
        file_name : Union[str,None], optional
            File name to save model weight to. When none is provide it will default to
            the model's name set in the meta data, by default None

        Raises
        ------
        IOError
            If file_name provided has a parent path that does not exist
        """
        if file_name is None:
            file_name = self.meta.name + ".pt"

        file_name = Path(file_name)
        if not file_name.parents[0].is_dir():
            raise IOError(
                f"Model checkpoint parent directory {file_name.parents[0]} not found"
            )

        torch.save(self.state_dict(), str(file_name))

    def load(self, file_name: Union[str, None] = None) -> None:
        """Simple utility for loading the model from checkpoint

        Parameters
        ----------
        file_name : Union[str,None], optional
            Checkpoint file name. When none is provide it will default to the model's
            name set in the meta data, by default None

        Raises
        ------
        IOError
            If file_name provided does not exist
        """
        if file_name is None:
            file_name = self.meta.name + ".pt"

        file_name = Path(file_name)
        if not file_name.exists():
            raise IOError(f"Model checkpoint {file_name} not found")

        model_dict = torch.load(file_name, map_location=self.device)
        self.load_state_dict(model_dict)

    @property
    def device(self) -> torch.device:
        """Get device model is on

        Returns
        -------
        torch.device
            PyTorch device
        """
        return self.device_buffer.device

    def num_parameters(self) -> int:
        """Gets the number of learnable parameters"""
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count


@torch.jit.script
def compl_mul1d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex-valued multiplication operation between two 1-dimensional
    tensors.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd1d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs complex multiplication of two 1-dimensional tensors 'a' and 'b', and then
    adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_mul1d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def compl_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex-valued multiplication operation between two 2-dimensional
    tensors.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs complex multiplication of two 2-dimensional tensors 'a' and 'b', and then
    adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script  # TODO remove
def _contract_localconv_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex local convolution operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script  # TODO remove
def _contract_blockconv_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex block convolution operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bim,imn->bin", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script  # TODO remove
def _contractadd_blockconv_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex block convolution operation between two tensors 'a' and 'b', and
    then adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(_contract_blockconv_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


# for the experimental layer
@torch.jit.script  # TODO remove
def compl_exp_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,xio->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_exp_muladd2d_fwd(  # TODO remove
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D complex multiplication operation between two tensors 'a' and 'b',
    and then adds a third tensor 'c'.
    """
    tmpcc = torch.view_as_complex(compl_exp_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def real_mul2d_fwd(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D real multiplication operation between two tensors 'a' and 'b'.
    """
    res = torch.einsum("bixy,io->boxy", a, b)
    return res


@torch.jit.script
def real_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a 2D real multiplication operation between two tensors 'a' and 'b', and
    then adds a third tensor 'c'.
    """
    res = real_mul2d_fwd(a, b) + c
    return res


# new contractions set to replace older ones. We use complex
@torch.jit.script
def _contract_diagonal(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex diagonal operation between two tensors 'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ioxy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_dhconv(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex Driscoll-Healy style convolution operation between two tensors
    'a' and 'b'.
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_diagonal(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex convolution operation between two tensors 'a' and 'b'
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ixy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_dhconv(
    a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex convolution operation between two tensors 'a' and 'b'
    """
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ix->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


import torch

import tensorly as tl

tl.set_backend("pytorch")
# from tensorly.plugins import use_opt_einsum
# use_opt_einsum('optimal')

from tltorch.factorized_tensors.core import FactorizedTensor

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(
    x, weight, separable=False, operator_type="diagonal"
):  # pragma: no cover
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

    if operator_type == "diagonal":
        pass
    elif operator_type == "block-diagonal":
        weight_syms.insert(-1, einsum_symbols[order + 1])
        out_syms[-1] = weight_syms[-2]
    elif operator_type == "dhconv":
        weight_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = "".join(x_syms) + "," + "".join(weight_syms) + "->" + "".join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return tl.einsum(eq, x, weight)


def _contract_cp(
    x, cp_weight, separable=False, operator_type="diagonal"
):  # pragma: no cover
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)

    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out

    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...

    if operator_type == "diagonal":
        pass
    elif operator_type == "block-diagonal":
        out_syms[-1] = einsum_symbols[order + 2]
        factor_syms += [out_syms[-1] + rank_sym]
    elif operator_type == "dhconv":
        factor_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = (
        x_syms + "," + rank_sym + "," + ",".join(factor_syms) + "->" + "".join(out_syms)
    )

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(
    x, tucker_weight, separable=False, operator_type="diagonal"
):  # pragma: no cover
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]  # x, y, ...

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        factor_syms += [
            xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])
        ]  # x, y, ...

    if operator_type == "diagonal":
        pass
    elif operator_type == "block-diagonal":
        raise NotImplementedError(
            f"Operator type {operator_type} not implemented for Tucker"
        )
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = (
        x_syms
        + ","
        + core_syms
        + ","
        + ",".join(factor_syms)
        + "->"
        + "".join(out_syms)
    )

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(
    x, tt_weight, separable=False, operator_type="diagonal"
):  # pragma: no cover
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size

    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)

    if operator_type == "diagonal":
        pass
    elif operator_type == "block-diagonal":
        weight_syms.insert(-1, einsum_symbols[order + 1])
        out_syms[-1] = weight_syms[-2]
    elif operator_type == "dhconv":
        weight_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    rank_syms = list(einsum_symbols[order + 2 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    return tl.einsum(eq, x, *tt_weight.factors)


# jitted PyTorch contractions:
def _contract_dense_pytorch(
    x, weight, separable=False, operator_type="diagonal"
):  # pragma: no cover

    # to cheat the fused optimizers convert to real here
    x = torch.view_as_real(x)

    if separable:
        if operator_type == "diagonal":
            x = _contract_sep_diagonal(x, weight)
        elif operator_type == "dhconv":
            x = _contract_sep_dhconv(x, weight)
        else:
            raise ValueError(f"Unkonw operator type {operator_type}")
    else:
        if operator_type == "diagonal":
            x = _contract_diagonal(x, weight)
        elif operator_type == "dhconv":
            x = _contract_dhconv(x, weight)
        else:
            raise ValueError(f"Unkonw operator type {operator_type}")

    # to cheat the fused optimizers convert to real here
    x = torch.view_as_complex(x)
    return x


def get_contract_fun(
    weight, implementation="reconstructed", separable=False, operator_type="diagonal"
):  # pragma: no cover
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input
        (factorized)

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense_pytorch
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower() == "complexdense" or weight.name.lower() == "dense":
                return _contract_dense
            elif weight.name.lower() == "complextucker":
                return _contract_tucker
            elif weight.name.lower() == "complextt":
                return _contract_tt
            elif weight.name.lower() == "complexcp":
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got {implementation=}, expected "reconstructed" or "factorized"'
        )


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
    tensor: an n-dimensional `torch.Tensor`
    mean: the mean of the normal distribution
    std: the standard deviation of the normal distribution
    a: the minimum cutoff value
    b: the maximum cutoff value
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)





@torch.jit.script
def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:  # pragma: no cover
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    This is the same as the DropConnect impl for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in
    a separate paper. See discussion:
        https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    We've opted for changing the layer and argument names to 'drop path' rather than
    mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual
    blocks).
    """

    def __init__(self, drop_prob=None):  # pragma: no cover
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):  # pragma: no cover
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    Divides the input image into patches and embeds them into a specified dimension
    using a convolutional layer.
    """

    def __init__(
        self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768
    ):  # pragma: no cover
        super(PatchEmbed, self).__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):  # pragma: no cover
        # gather input
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


class MLP(nn.Module):
    """
    Basic CNN with support for gradient checkpointing
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        output_bias=True,
        drop_rate=0.0,
        checkpointing=0,
    ):  # pragma: no cover
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        act = act_layer()
        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        if drop_rate > 0.0:
            drop = nn.Dropout(drop_rate)
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
        else:
            self.fwd = nn.Sequential(fc1, act, fc2)

        # by default, all weights are shared

    @torch.jit.ignore
    def checkpoint_forward(self, x):  # pragma: no cover
        """Forward method with support for gradient checkpointing"""
        return checkpoint(self.fwd, x)

    def forward(self, x):  # pragma: no cover
        if self.checkpointing >= 2:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)


class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(RealFFT2, self).__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        # self.num_batches = 1
        assert self.lmax % 2 == 0

    def forward(self, x):  # pragma: no cover
        y = self.fft_handle(x, (self.nlat, self.nlon), (-2, -1), "ortho")

        if self.truncate:
            y = torch.cat(
                (
                    y[..., : math.ceil(self.lmax / 2), : self.mmax],
                    y[..., -math.floor(self.lmax / 2) :, : self.mmax],
                ),
                dim=-2,
            )

        return y


class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(InverseRealFFT2, self).__init__()

        # use local FFT here
        self.ifft_handle = torch.fft.irfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):  # pragma: no cover
        out = self.ifft_handle(x, (self.nlat, self.nlon), (-2, -1), "ortho")

        return out



class SpectralConvS2(nn.Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on
    the two-sphere S2 using the Spherical Harmonic Transforms in torch-harmonics, but
    supports convolutions on the periodic domain via the RealFFT2 and InverseRealFFT2
    wrappers.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        scale="auto",
        operator_type="diagonal",
        rank=0.2,
        factorization=None,
        separable=False,
        decomposition_kwargs=dict(),
        bias=False,
        use_tensorly=True,
    ):  # pragma: no cover
        super(SpectralConvS2, self).__init__()

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (
            (self.forward_transform.nlat != self.inverse_transform.nlat)
            or (self.forward_transform.nlon != self.inverse_transform.nlon)
            or (self.forward_transform.grid != self.inverse_transform.grid)
        )

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = "Dense"  # No factorization

        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        # remember factorization details
        self.operator_type = operator_type
        self.rank = rank
        self.factorization = factorization
        self.separable = separable

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [in_channels]

        if not self.separable:
            weight_shape += [out_channels]

        if isinstance(self.inverse_transform, DistributedInverseRealSHT):
            self.modes_lat_local = self.inverse_transform.lmax_local
            self.modes_lon_local = self.inverse_transform.mmax_local
            self.lpad_local = self.inverse_transform.lpad_local
            self.mpad_local = self.inverse_transform.mpad_local
        else:
            self.modes_lat_local = self.modes_lat
            self.modes_lon_local = self.modes_lon
            self.lpad = 0
            self.mpad = 0

        # padded weights
        # if self.operator_type == 'diagonal':
        #     weight_shape += [self.modes_lat_local+self.lpad_local, self.modes_lon_local+self.mpad_local]
        # elif self.operator_type == 'dhconv':
        #     weight_shape += [self.modes_lat_local+self.lpad_local]
        # else:
        #     raise ValueError(f"Unsupported operator type f{self.operator_type}")

        # unpadded weights
        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat_local, self.modes_lon_local]
        elif self.operator_type == "dhconv":
            weight_shape += [self.modes_lat_local]
        else:
            raise ValueError(f"Unsupported operator type f{self.operator_type}")

        if use_tensorly:
            # form weight tensors
            self.weight = FactorizedTensor.new(
                weight_shape,
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=False,
                **decomposition_kwargs,
            )
            # initialization of weights
            self.weight.normal_(0, scale)
        else:
            assert factorization == "ComplexDense"
            self.weight = nn.Parameter(scale * torch.randn(*weight_shape, 2))
            if self.operator_type == "dhconv":
                self.weight.is_shared_mp = ["matmul", "w"]
            else:
                self.weight.is_shared_mp = ["matmul"]

        # get the contraction handle
        self._contract = get_contract_fun(
            self.weight, implementation="factorized", separable=separable
        )

        if bias:
            self.bias = nn.Parameter(scale * torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):  # pragma: no cover

        dtype = x.dtype
        residual = x
        x = x.float()
        B, C, H, W = x.shape

        with amp.autocast(enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # approach with unpadded weights
        xp = torch.zeros_like(x)
        xp[..., : self.modes_lat_local, : self.modes_lon_local] = self._contract(
            x[..., : self.modes_lat_local, : self.modes_lon_local],
            self.weight,
            separable=self.separable,
            operator_type=self.operator_type,
        )
        x = xp.contiguous()

        # # approach with padded weights
        # x = self._contract(x, self.weight, separable=self.separable, operator_type=self.operator_type)
        # x = x.contiguous()

        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x, residual


class LocalConvS2(nn.Module):
    """
    S2 Convolution according to Driscoll & Healy
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        nradius=120,
        scale="auto",
        bias=False,
    ):  # pragma: no cover
        super(LocalConvS2, self).__init__()

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nradius = nradius

        self.forward_transform = forward_transform
        self.zonal_transform = th.RealSHT(
            forward_transform.nlat,
            1,
            lmax=forward_transform.lmax,
            mmax=1,
            grid=forward_transform.grid,
        ).float()
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax
        self.output_dims = (self.inverse_transform.nlat, self.inverse_transform.nlon)

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, nradius, 1)
        )

        self._contract = _contract_localconv_fwd

        if bias:
            self.bias = nn.Parameter(
                scale * torch.randn(1, out_channels, *self.output_dims)
            )

    def forward(self, x):  # pragma: no cover

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        with amp.autocast(enabled=False):
            f = torch.zeros(
                (self.in_channels, self.out_channels, H, 1),
                dtype=x.dtype,
                device=x.device,
            )
            f[..., : self.nradius, :] = self.weight

            x = self.forward_transform(x)
            f = self.zonal_transform(f)[..., :, 0]

            x = torch.view_as_real(x)
            f = torch.view_as_real(f)

        x = self._contract(x, f)
        x = x.contiguous()

        x = torch.view_as_complex(x)

        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x


class SpectralAttentionS2(nn.Module):
    """
    Spherical non-linear FNO layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        scale="auto",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        if scale == "auto":
            self.scale = 1 / (embed_dim * embed_dim)

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        hidden_size = int(hidden_size_factor * self.embed_dim)

        if operator_type == "diagonal":
            self.mul_add_handle = compl_muladd2d_fwd
            self.mul_handle = compl_mul2d_fwd

            # weights
            w = [self.scale * torch.randn(self.embed_dim, hidden_size, 2)]
            for l in range(1, self.spectral_layers):
                w.append(self.scale * torch.randn(hidden_size, hidden_size, 2))
            self.w = nn.ParameterList(w)

            self.wout = nn.Parameter(
                self.scale * torch.randn(hidden_size, self.embed_dim, 2)
            )

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        elif operator_type == "l-dependant":

            self.mul_add_handle = compl_exp_muladd2d_fwd
            self.mul_handle = compl_exp_mul2d_fwd

            # weights
            w = [
                self.scale * torch.randn(self.modes_lat, self.embed_dim, hidden_size, 2)
            ]
            for l in range(1, self.spectral_layers):
                w.append(
                    self.scale
                    * torch.randn(self.modes_lat, hidden_size, hidden_size, 2)
                )
            self.w = nn.ParameterList(w)

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.wout = nn.Parameter(
                self.scale * torch.randn(self.modes_lat, hidden_size, self.embed_dim, 2)
            )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        else:
            raise ValueError("Unknown operator type")

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        if self.operator_type == "block-separable":
            x = x.permute(0, 3, 1, 2)

        xr = torch.view_as_real(x)

        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[l], self.b[l])
            else:
                xr = self.mul_handle(xr, self.w[l])
            xr = torch.view_as_complex(xr)
            xr = self.activations[l](xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        # final MLP
        x = self.mul_handle(xr, self.wout)

        x = torch.view_as_complex(x)

        if self.operator_type == "block-separable":
            x = x.permute(0, 2, 3, 1)

        return x

    def forward(self, x):  # pragma: no cover

        dtype = x.dtype
        residual = x
        x = x.to(torch.float32)

        # FWD transform
        with amp.autocast(enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        x = x.contiguous()
        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        # cast back to initial precision
        x = x.to(dtype)

        return x, residual


class RealSpectralAttentionS2(nn.Module):
    """
    Non-linear SFNO layer using a real-valued NN instead of a complex one
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        scale="auto",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(RealSpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        if scale == "auto":
            self.scale = 1 / (embed_dim * embed_dim)

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        hidden_size = int(hidden_size_factor * self.embed_dim * 2)

        self.mul_add_handle = real_muladd2d_fwd
        self.mul_handle = real_mul2d_fwd

        # weights
        w = [self.scale * torch.randn(2 * self.embed_dim, hidden_size)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(hidden_size, hidden_size))
        self.w = nn.ParameterList(w)

        self.wout = nn.Parameter(
            self.scale * torch.randn(hidden_size, 2 * self.embed_dim)
        )

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(hidden_size, 1, 1)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.activations = nn.ModuleList([])
        for l in range(0, self.spectral_layers):
            self.activations.append(nn.ReLU())

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        xr = torch.view_as_real(x)
        xr = xr.permute(0, 1, 4, 2, 3).reshape(B, C * 2, H, W)

        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[l], self.b[l])
            else:
                xr = self.mul_handle(xr, self.w[l])
            xr = self.activations[l](xr)
            xr = self.drop(xr)

        # final MLP
        xr = self.mul_handle(xr, self.wout)

        xr = xr.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2)

        x = torch.view_as_complex(xr)

        return x

    def forward(self, x):  # pragma: no cover

        dtype = x.dtype
        x = x.to(torch.float32)

        # FWD transform
        with amp.autocast(enabled=False):
            x = self.forward_transform(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        # cast back to initial precision
        x = x.to(dtype)

        return x



import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch._utils import _flatten_dense_tensors


def get_memory_format(tensor):  # pragma: no cover
    """Helper routine to get the memory format"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format


def sync_params(model, mode="broadcast"):  # pragma: no cover
    """Helper routine to ensure shared weights are the same after initialization"""

    non_singleton_group_names = [
        x
        for x in comm.get_names()
        if (comm.get_size(x) > 1) and not (x in ["data", "model", "spatial"])
    ]

    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():

            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = non_singleton_group_names.copy()

            for comm_group in param.is_shared_mp:
                if comm.get_size(comm_group) > 1:
                    if mode == "broadcast":
                        tlist = [
                            torch.empty_like(param)
                            for x in range(comm.get_size(comm_group))
                        ]
                        tlist[comm.get_rank(comm_group)] = param
                        # gather all weights in the comm group
                        dist.all_gather(tlist, param, group=comm.get_group(comm_group))
                        # use weight of rank 0
                        # important to use copy here otherwise the handle gets detaches from the optimizer
                        param.copy_(tlist[0])
                    elif mode == "mean":
                        # coalesced = _flatten_dense_tensors(param)
                        dist.all_reduce(
                            param,
                            op=dist.ReduceOp.AVG,
                            group=comm.get_group(comm_group),
                            async_op=False,
                        )
                        # param.copy_(coalesced)
                    else:
                        raise ValueError(f"Unknown weight synchronization mode {mode}")


def pad_helper(tensor, dim, new_size, mode="zero"):  # pragma: no cover
    """Helper routine to pad a tensor along a given dimension"""
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode="constant", value=0.0)

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad[lhs_slice] = torch.flip(
            torch.conj(tensor_pad[rhs_slice]), dims=[dim]
        )

    return tensor_pad


def truncate_helper(tensor, dim, new_size):  # pragma: no cover
    """Helper routine to truncate a tensor along a given dimension"""
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[output_slice].contiguous(memory_format=input_format)

    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):  # pragma: no cover
    """Helper routine to split a tensor along a given dimension"""
    assert (
        dim < tensor.dim()
    ), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (
        tensor.shape[dim] % num_chunks == 0
    ), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                  {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = torch.split(tensor, chunk_size, dim=dim)

    return tensor_list


# distributed primitives
def _transpose(tensor, dim0, dim1, group=None, async_op=False):  # pragma: no cover
    """Transpose a tensor across model parallel group."""
    # get input format
    input_format = get_memory_format(tensor)

    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [
        y.contiguous(memory_format=input_format)
        for y in torch.split(tensor, split_size, dim=dim0)
    ]
    x_recv = [torch.empty_like(x_send[0]) for _ in range(comm_size)]

    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)

    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):  # pragma: no cover
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_


def _split(input_, dim_, group=None):  # pragma: no cover
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def _gather(input_, dim_, group=None):  # pragma: no cover
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    assert (
        dim_ < input_.dim()
    ), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    input_ = input_.contiguous(memory_format=input_format)
    tensor_list = [torch.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    dist.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)

    return output




import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


# torch utils
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# generalized
class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):  # pragma: no cover
        """symbolic method"""
        return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):  # pragma: no cover
        ctx.comm_id = comm_id_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if comm.is_distributed(ctx.comm_id):  # pragma: no cover
            return _reduce(grad_output, group=comm.get_group(ctx.comm_id)), None
        else:
            return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):  # pragma: no cover
        """symbolic method"""
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):  # pragma: no cover
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_):  # pragma: no cover
        """symbolic method"""
        return _split(input_, dim_, group=comm.get_group(comm_id_))

    @staticmethod
    def forward(ctx, input_, dim_, comm_id_):  # pragma: no cover
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        if comm.is_distributed(comm_id_):
            return _split(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        if comm.is_distributed(ctx.comm_id):
            return (
                _gather(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)),
                None,
                None,
            )
        else:
            return grad_output, None, None


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_):  # pragma: no cover
        """"""
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, dim_, comm_id_):  # pragma: no cover
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        if comm.is_distributed(ctx.comm_id):
            return (
                _split(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)),
                None,
                None,
            )
        else:
            return grad_output, None, None


# -----------------
# Helper functions.
# -----------------
# matmul parallel
def copy_to_matmul_parallel_region(input_):  # pragma: no cover
    """copy helper"""
    return _CopyToParallelRegion.apply(input_, "matmul")


def reduce_from_matmul_parallel_region(input_):  # pragma: no cover
    """reduce helper"""
    return _ReduceFromParallelRegion.apply(input_, "matmul")


def scatter_to_matmul_parallel_region(input_, dim):  # pragma: no cover
    """scatter helper"""
    return _ScatterToParallelRegion.apply(input_, dim, "matmul")


def gather_from_matmul_parallel_region(input_, dim):  # pragma: no cover
    """gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, "matmul")


# general
def reduce_from_parallel_region(input_, comm_name):  # pragma: no cover
    """reduce helper"""
    return _ReduceFromParallelRegion.apply(input_, comm_name)


def scatter_to_parallel_region(input_, dim, comm_name):  # pragma: no cover
    """scatter helper"""
    return _ScatterToParallelRegion.apply(input_, dim, comm_name)


def gather_from_parallel_region(input_, dim, comm_name):  # pragma: no cover
    """gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, comm_name)


# def gather_within_matmul_parallel_region(input_, dim):
#    return _GatherWithinMatmulParallelRegion.apply(input_, dim, "matmul")

# spatial parallel
def copy_to_spatial_parallel_region(input_):  # pragma: no cover
    """copy helper"""
    return _CopyToParallelRegion.apply(input_, "spatial")


def scatter_to_spatial_parallel_region(input_, dim):  # pragma: no cover
    """scatter helper"""
    return _ScatterToParallelRegion.apply(input_, dim, "spatial")


def gather_from_spatial_parallel_region(input_, dim):  # pragma: no cover
    """gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, "spatial")


# handler for additional gradient reductions
# helper for gradient reduction across channel parallel ranks
def init_gradient_reduction_hooks(
    model,
    device_ids,
    output_device,
    bucket_cap_mb=25,
    broadcast_buffers=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=False,
):  # pragma: no cover
    """
    Initialize gradient reduction hooks for a given model.
    """

    # early exit if we are not in a distributed setting:
    if not dist.is_initialized():
        return model

    # set this to false in init and then find out if we can use it:
    need_hooks = False
    ddp_group = comm.get_group("data")

    # this is the trivial case
    if comm.get_size("model") == 1:
        # the simple case, we can just continue then
        ddp_group = None
    else:
        # check if there are shared weights, otherwise we can skip
        non_singleton_group_names = [
            x
            for x in comm.get_names()
            if (comm.get_size(x) > 1) and not (x in ["data", "model", "spatial"])
        ]
        num_shared = {x: 0 for x in non_singleton_group_names}
        num_parameters = 0

        # count parameters and reduction groups
        for param in model.parameters():

            # if it does not have any annotation, we assume it is shared between all groups
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = non_singleton_group_names.copy()

            # check remaining groups
            for group in non_singleton_group_names:
                if group in param.is_shared_mp:
                    num_shared[group] += 1
            num_parameters += 1

        # group without data:
        num_param_shared_model = [v for k, v in num_shared.items()]
        if not num_param_shared_model:
            num_shared_model = 0
        else:
            num_shared_model = sum(num_param_shared_model)

        # if all parameters are just data shared and not additionally shared orthogonally to that, we can use DDP
        if num_shared_model == 0:
            ddp_group = None

        elif all([(x == num_parameters) for x in num_param_shared_model]):
            # in this case, we just need to register a backward hook to multiply the gradients according to the multiplicity:
            print(
                "Setting up gradient hooks to account for shared parameter multiplicity"
            )
            for param in model.parameters():
                param.register_hook(lambda grad: grad * float(comm.get_size("model")))

            ddp_group = None
        else:
            ddp_group = comm.get_group("data")  # double check if this is correct
            broadcast_buffers = False
            need_hooks = True

    # we can set up DDP and exit here
    print("Setting up DDP communication hooks")
    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        bucket_cap_mb=bucket_cap_mb,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=ddp_group,
    )
    if not need_hooks:
        return model

    print("Setting up custom communication hooks")

    # define comm hook:
    def reduction_comm_hook(
        state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:  # pragma: no cover
        """reduction comm hook"""

        # allreduce everything first:
        buff = bucket.buffer()

        # get future for allreduce
        fut = dist.all_reduce(
            buff, op=dist.ReduceOp.AVG, group=comm.get_group("data"), async_op=True
        ).get_future()

        # get grads for shared weights
        params = bucket.parameters()

        def grad_reduction(fut, grads, group):
            """reduce remaining gradients"""
            coalesced = _flatten_dense_tensors(grads)
            dist.all_reduce(
                coalesced,
                op=dist.ReduceOp.SUM,
                group=comm.get_group(group),
                async_op=False,
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

            return bucket.buffer()

        for group in non_singleton_group_names:
            if group == "data":
                continue

            grads = []
            for p in params:
                if group in p.is_shared_mp:
                    grads.append(p.grad.data)

            if not grads:
                continue

            # append the new reduction functions
            fut = fut.then(lambda x: grad_reduction(x, grads=grads, group=group))
            # fut = fut.then(lambda x: grad_copy(x, grads=grads))

        ## chain it together
        # for redfut, copyfut in zip(redfunc, copyfunc):
        #    fut = fut.then(redfut).then(copyfut)

        return fut

    # register model comm hook
    model.register_comm_hook(state=None, hook=reduction_comm_hook)

    return model




import torch
import torch.nn as nn
import torch.nn.functional as F

class distributed_transpose_w(torch.autograd.Function):
    """Distributed transpose"""

    @staticmethod
    def forward(ctx, x, dim):  # pragma: no cover
        xlist, _ = _transpose(x, dim[0], dim[1], group=comm.get_group("w"))
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):  # pragma: no cover
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=comm.get_group("w"))
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None


class distributed_transpose_h(torch.autograd.Function):
    """Distributed transpose"""

    @staticmethod
    def forward(ctx, x, dim):  # pragma: no cover
        xlist, _ = _transpose(x, dim[0], dim[1], group=comm.get_group("h"))
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):  # pragma: no cover
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=comm.get_group("h"))
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None


class DistributedRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(DistributedRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        # frequency paddings
        ldist = (self.lmax + self.comm_size_h - 1) // self.comm_size_h
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_w - 1) // self.comm_size_w
        self.mpad = mdist * self.comm_size_w - self.mmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        # we need to ensure that we can split the channels evenly
        assert x.shape[1] % self.comm_size_h == 0
        assert x.shape[1] % self.comm_size_w == 0

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            xt = distributed_transpose_w.apply(x, (1, -1))
        else:
            xt = x

        # do first FFT
        xtf = torch.fft.rfft(xt, n=self.nlon, dim=-1, norm="ortho")

        # truncate
        xtft = xtf[..., : self.mmax]

        # pad the dim to allow for splitting
        xtfp = F.pad(xtft, [0, self.mpad], mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            y = distributed_transpose_w.apply(xtfp, (-1, 1))
        else:
            y = xtfp

        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            yt = distributed_transpose_h.apply(y, (1, -2))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        # ytt = yt[..., :self.nlat, :]

        # do second FFT:
        yo = torch.fft.fft(yt, n=self.nlat, dim=-2, norm="ortho")

        # pad if required, truncation is implicit
        yop = F.pad(yo, [0, 0, 0, self.lpad], mode="constant")

        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(yop, (-2, 1))
        else:
            y = yop

        return y


class DistributedInverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(DistributedInverseRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_h - 1) // self.comm_size_h
        self.latpad = latdist * self.comm_size_h - self.nlat
        londist = (self.nlon + self.comm_size_w - 1) // self.comm_size_w
        self.lonpad = londist * self.comm_size_w - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_h - 1) // self.comm_size_h
        self.lpad = ldist * self.comm_size_h - self.lmax
        mdist = (self.mmax + self.comm_size_w - 1) // self.comm_size_w
        self.mpad = mdist * self.comm_size_w - self.mmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        # we need to ensure that we can split the channels evenly
        assert x.shape[1] % self.comm_size_h == 0
        assert x.shape[1] % self.comm_size_w == 0

        # transpose: after that, channels are split, l is local:
        if self.comm_size_h > 1:
            xt = distributed_transpose_h.apply(x, (1, -2))
        else:
            xt = x

        # truncate
        xtt = xt[..., : self.lmax, :]

        # do first fft
        xf = torch.fft.ifft(xtt, n=self.nlat, dim=-2, norm="ortho")

        # transpose: after this, l is split and channels are local
        xfp = F.pad(xf, [0, 0, 0, self.latpad])

        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(xfp, (-2, 1))
        else:
            y = xfp

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            yt = distributed_transpose_w.apply(y, (1, -1))
        else:
            yt = y

        # truncate
        ytt = yt[..., : self.mmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(ytt, n=self.nlon, dim=-1, norm="ortho")

        # pad before we transpose back
        xp = F.pad(x, [0, self.lonpad])

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            out = distributed_transpose_w.apply(xp, (-1, 1))
        else:
            out = xp

        return out


# more complicated layers
class DistributedMLP(nn.Module):
    """Distributed MLP layer"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        output_bias=True,
        act_layer=nn.GELU,
        drop_rate=0.0,
        checkpointing=False,
    ):  # pragma: no cover

        super(DistributedMLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # get effective embedding size:
        comm_size = comm.get_size("matmul")
        assert (
            hidden_features % comm_size == 0
        ), "Error, hidden_features needs to be divisible by matmul_parallel_size"
        hidden_features_local = hidden_features // comm_size

        # first set of hp
        self.w1 = nn.Parameter(torch.ones(hidden_features_local, in_features, 1, 1))
        self.b1 = nn.Parameter(torch.zeros(hidden_features_local))

        # second set of hp
        self.w2 = nn.Parameter(torch.ones(out_features, hidden_features_local, 1, 1))

        if output_bias:
            self.b2 = nn.Parameter(torch.zeros(out_features))

        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop_rate > 0.0 else nn.Identity()

        # the weights are shared spatially
        self.w1.is_shared_mp = ["h", "w"]
        self.b1.is_shared_mp = ["h", "w"]
        self.w2.is_shared_mp = ["h", "w"]
        if output_bias:
            self.b2.is_shared_mp = [
                "matmul",
                "h",
                "w",
            ]  # this one is shared between all ranks

        # init weights
        self._init_weights()

    def _init_weights(self):  # pragma: no cover
        trunc_normal_(self.w1, std=0.02)
        nn.init.constant_(self.b1, 0.0)
        trunc_normal_(self.w2, std=0.02)
        if hasattr(self, "b2"):
            nn.init.constant_(self.b2, 0.0)

    def fwd(self, x):  # pragma: no cover
        """Forward function."""
        # we need to prepare paralellism here
        # spatial parallelism
        x = scatter_to_spatial_parallel_region(x, dim=-1)

        # prepare the matmul parallel part
        x = copy_to_matmul_parallel_region(x)

        # do the mlp
        x = F.conv2d(x, self.w1, bias=self.b1)
        x = self.act(x)
        x = self.drop(x)
        x = F.conv2d(x, self.w2, bias=None)
        x = reduce_from_matmul_parallel_region(x)
        if hasattr(self, "b2"):
            x = x + torch.reshape(self.b2, (1, -1, 1, 1))
        x = self.drop(x)

        # gather from spatial parallel region
        x = gather_from_spatial_parallel_region(x, dim=-1)

        return x

    @torch.jit.ignore
    def _checkpoint_forward(self, x):  # pragma: no cover
        return checkpoint(self.fwd, x)

    def forward(self, x):  # pragma: no cover
        if self.checkpointing:
            return self._checkpoint_forward(x)
        else:
            return self.fwd(x)


class DistributedPatchEmbed(nn.Module):
    """Distributed patch embedding layer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=True,
    ):  # pragma: no cover

        super(DistributedPatchEmbed, self).__init__()

        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        spatial_comm_size = comm.get_size("spatial")

        # compute parameters
        assert (
            img_size[1] // patch_size[1]
        ) % spatial_comm_size == 0, (
            "Error, make sure that the spatial comm size evenly divides patched W"
        )
        num_patches = ((img_size[1] // patch_size[1]) // spatial_comm_size) * (
            img_size[0] // patch_size[0]
        )
        self.img_size = (img_size[0], img_size[1] // spatial_comm_size)
        self.patch_size = patch_size
        self.num_patches = num_patches

        # get effective embedding size:
        if self.output_parallel:
            assert (
                embed_dim % matmul_comm_size == 0
            ), "Error, the embed_dim needs to be divisible by matmul_parallel_size"
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(
            in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size
        )

        # make sure we reduce them across rank
        self.proj.weight.is_shared_mp = ["h", "w"]
        self.proj.bias.is_shared_mp = ["h", "w"]

    def forward(self, x):  # pragma: no cover
        if self.input_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)

        if self.output_parallel:
            x = copy_to_matmul_parallel_region(x)

        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


@torch.jit.script
def compl_mul_add_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """complex multiplication and addition"""
    tmp = torch.einsum("bkixys,kiot->stbkoxy", a, b)
    res = (
        torch.stack(
            [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
        )
        + c
    )
    return res


@torch.jit.script
def compl_mul_add_fwd_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Performs a complex multiplication and addition operation on three tensors"""
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    tmp = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = tmp + cc
    return torch.view_as_real(res)


class DistributedAFNO2Dv2(nn.Module):
    """Distributed AFNO"""

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
        use_complex_kernels=False,
    ):  # pragma: no cover
        """Distributed AFNO2Dv2"""
        super(DistributedAFNO2Dv2, self).__init__()
        assert (
            hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        self.spatial_comm_size = comm.get_size("spatial")

        # select fft function handles
        if self.spatial_comm_size > 1:
            self.fft_handle = distributed_rfft2.apply
            self.ifft_handle = distributed_irfft2.apply
        else:
            self.fft_handle = torch.fft.rfft2
            self.ifft_handle = torch.fft.irfft2

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        assert (
            self.num_blocks % matmul_comm_size == 0
        ), "Error, num_blocks needs to be divisible by matmul_parallel_size"
        self.num_blocks_local = self.num_blocks // matmul_comm_size
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.mult_handle = (
            compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd
        )

        # model paralellism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # new
        # these weights need to be synced across all spatial ranks!
        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size,
                self.block_size * self.hidden_size_factor,
                2,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                1,
                1,
                2,
            )
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                self.block_size,
                2,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks_local, self.block_size, 1, 1, 2)
        )

        # make sure we reduce them across rank
        self.w1.is_shared_mp = ["h", "w"]
        self.b1.is_shared_mp = ["h", "w"]
        self.w2.is_shared_mp = ["h", "w"]
        self.b2.is_shared_mp = ["h", "w"]

    def forward(self, x):  # pragma: no cover
        if not self.input_is_matmul_parallel:
            # distribute data
            x = scatter_to_matmul_parallel_region(x, dim=1)

        # bias
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W_local = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        H_local = H // self.spatial_comm_size
        W = W_local * self.spatial_comm_size
        x = self.fft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.view(B, self.num_blocks_local, self.block_size, H_local, W // 2 + 1)

        # new
        x = torch.view_as_real(x)
        o2 = torch.zeros(x.shape, device=x.device)

        o1 = F.relu(
            self.mult_handle(
                x[
                    :,
                    :,
                    :,
                    total_modes - kept_modes : total_modes + kept_modes,
                    :kept_modes,
                    :,
                ],
                self.w1,
                self.b1,
            )
        )
        o2[
            :, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :
        ] = self.mult_handle(o1, self.w2, self.b2)

        # finalize
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H_local, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.type(dtype) + bias

        # gather
        if not self.output_is_matmul_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)

        return x


# generalized
class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):  # pragma: no cover
        """symbolic method"""
        return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):  # pragma: no cover
        ctx.comm_id = comm_id_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if comm.is_distributed(ctx.comm_id):  # pragma: no cover
            return _reduce(grad_output, group=comm.get_group(ctx.comm_id)), None
        else:
            return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):  # pragma: no cover
        """symbolic method"""
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):  # pragma: no cover
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_):  # pragma: no cover
        """symbolic method"""
        return _split(input_, dim_, group=comm.get_group(comm_id_))

    @staticmethod
    def forward(ctx, input_, dim_, comm_id_):  # pragma: no cover
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        if comm.is_distributed(comm_id_):
            return _split(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        if comm.is_distributed(ctx.comm_id):
            return (
                _gather(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)),
                None,
                None,
            )
        else:
            return grad_output, None, None


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_):  # pragma: no cover
        """"""
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, dim_, comm_id_):  # pragma: no cover
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        if comm.is_distributed(ctx.comm_id):
            return (
                _split(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)),
                None,
                None,
            )
        else:
            return grad_output, None, None


# -----------------
# Helper functions.
# -----------------
# matmul parallel
def copy_to_matmul_parallel_region(input_):  # pragma: no cover
    """copy helper"""
    return _CopyToParallelRegion.apply(input_, "matmul")


def reduce_from_matmul_parallel_region(input_):  # pragma: no cover
    """reduce helper"""
    return _ReduceFromParallelRegion.apply(input_, "matmul")


def scatter_to_matmul_parallel_region(input_, dim):  # pragma: no cover
    """scatter helper"""
    return _ScatterToParallelRegion.apply(input_, dim, "matmul")


def gather_from_matmul_parallel_region(input_, dim):  # pragma: no cover
    """gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, "matmul")


# general
def reduce_from_parallel_region(input_, comm_name):  # pragma: no cover
    """reduce helper"""
    return _ReduceFromParallelRegion.apply(input_, comm_name)


def scatter_to_parallel_region(input_, dim, comm_name):  # pragma: no cover
    """scatter helper"""
    return _ScatterToParallelRegion.apply(input_, dim, comm_name)


def gather_from_parallel_region(input_, dim, comm_name):  # pragma: no cover
    """gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, comm_name)


# def gather_within_matmul_parallel_region(input_, dim):
#    return _GatherWithinMatmulParallelRegion.apply(input_, dim, "matmul")

# spatial parallel
def copy_to_spatial_parallel_region(input_):  # pragma: no cover
    """copy helper"""
    return _CopyToParallelRegion.apply(input_, "spatial")


def scatter_to_spatial_parallel_region(input_, dim):  # pragma: no cover
    """scatter helper"""
    return _ScatterToParallelRegion.apply(input_, dim, "spatial")


def gather_from_spatial_parallel_region(input_, dim):  # pragma: no cover
    """gather helper"""
    return _GatherFromParallelRegion.apply(input_, dim, "spatial")


# handler for additional gradient reductions
# helper for gradient reduction across channel parallel ranks
def init_gradient_reduction_hooks(
    model,
    device_ids,
    output_device,
    bucket_cap_mb=25,
    broadcast_buffers=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=False,
):  # pragma: no cover
    """
    Initialize gradient reduction hooks for a given model.
    """

    # early exit if we are not in a distributed setting:
    if not dist.is_initialized():
        return model

    # set this to false in init and then find out if we can use it:
    need_hooks = False
    ddp_group = comm.get_group("data")

    # this is the trivial case
    if comm.get_size("model") == 1:
        # the simple case, we can just continue then
        ddp_group = None
    else:
        # check if there are shared weights, otherwise we can skip
        non_singleton_group_names = [
            x
            for x in comm.get_names()
            if (comm.get_size(x) > 1) and not (x in ["data", "model", "spatial"])
        ]
        num_shared = {x: 0 for x in non_singleton_group_names}
        num_parameters = 0

        # count parameters and reduction groups
        for param in model.parameters():

            # if it does not have any annotation, we assume it is shared between all groups
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = non_singleton_group_names.copy()

            # check remaining groups
            for group in non_singleton_group_names:
                if group in param.is_shared_mp:
                    num_shared[group] += 1
            num_parameters += 1

        # group without data:
        num_param_shared_model = [v for k, v in num_shared.items()]
        if not num_param_shared_model:
            num_shared_model = 0
        else:
            num_shared_model = sum(num_param_shared_model)

        # if all parameters are just data shared and not additionally shared orthogonally to that, we can use DDP
        if num_shared_model == 0:
            ddp_group = None

        elif all([(x == num_parameters) for x in num_param_shared_model]):
            # in this case, we just need to register a backward hook to multiply the gradients according to the multiplicity:
            print(
                "Setting up gradient hooks to account for shared parameter multiplicity"
            )
            for param in model.parameters():
                param.register_hook(lambda grad: grad * float(comm.get_size("model")))

            ddp_group = None
        else:
            ddp_group = comm.get_group("data")  # double check if this is correct
            broadcast_buffers = False
            need_hooks = True

    # we can set up DDP and exit here
    print("Setting up DDP communication hooks")
    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        bucket_cap_mb=bucket_cap_mb,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=ddp_group,
    )
    if not need_hooks:
        return model

    print("Setting up custom communication hooks")

    # define comm hook:
    def reduction_comm_hook(
        state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:  # pragma: no cover
        """reduction comm hook"""

        # allreduce everything first:
        buff = bucket.buffer()

        # get future for allreduce
        fut = dist.all_reduce(
            buff, op=dist.ReduceOp.AVG, group=comm.get_group("data"), async_op=True
        ).get_future()

        # get grads for shared weights
        params = bucket.parameters()

        def grad_reduction(fut, grads, group):
            """reduce remaining gradients"""
            coalesced = _flatten_dense_tensors(grads)
            dist.all_reduce(
                coalesced,
                op=dist.ReduceOp.SUM,
                group=comm.get_group(group),
                async_op=False,
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

            return bucket.buffer()

        for group in non_singleton_group_names:
            if group == "data":
                continue

            grads = []
            for p in params:
                if group in p.is_shared_mp:
                    grads.append(p.grad.data)

            if not grads:
                continue

            # append the new reduction functions
            fut = fut.then(lambda x: grad_reduction(x, grads=grads, group=group))
            # fut = fut.then(lambda x: grad_copy(x, grads=grads))

        ## chain it together
        # for redfut, copyfut in zip(redfunc, copyfunc):
        #    fut = fut.then(redfut).then(copyfut)

        return fut

    # register model comm hook
    model.register_comm_hook(state=None, hook=reduction_comm_hook)

    return model





class distributed_transpose_w(torch.autograd.Function):
    """Distributed transpose"""

    @staticmethod
    def forward(ctx, x, dim):  # pragma: no cover
        xlist, _ = _transpose(x, dim[0], dim[1], group=comm.get_group("w"))
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):  # pragma: no cover
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=comm.get_group("w"))
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None


class distributed_transpose_h(torch.autograd.Function):
    """Distributed transpose"""

    @staticmethod
    def forward(ctx, x, dim):  # pragma: no cover
        xlist, _ = _transpose(x, dim[0], dim[1], group=comm.get_group("h"))
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):  # pragma: no cover
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=comm.get_group("h"))
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None


class DistributedRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(DistributedRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        # frequency paddings
        ldist = (self.lmax + self.comm_size_h - 1) // self.comm_size_h
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_w - 1) // self.comm_size_w
        self.mpad = mdist * self.comm_size_w - self.mmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        # we need to ensure that we can split the channels evenly
        assert x.shape[1] % self.comm_size_h == 0
        assert x.shape[1] % self.comm_size_w == 0

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            xt = distributed_transpose_w.apply(x, (1, -1))
        else:
            xt = x

        # do first FFT
        xtf = torch.fft.rfft(xt, n=self.nlon, dim=-1, norm="ortho")

        # truncate
        xtft = xtf[..., : self.mmax]

        # pad the dim to allow for splitting
        xtfp = F.pad(xtft, [0, self.mpad], mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            y = distributed_transpose_w.apply(xtfp, (-1, 1))
        else:
            y = xtfp

        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            yt = distributed_transpose_h.apply(y, (1, -2))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        # ytt = yt[..., :self.nlat, :]

        # do second FFT:
        yo = torch.fft.fft(yt, n=self.nlat, dim=-2, norm="ortho")

        # pad if required, truncation is implicit
        yop = F.pad(yo, [0, 0, 0, self.lpad], mode="constant")

        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(yop, (-2, 1))
        else:
            y = yop

        return y


class DistributedInverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(DistributedInverseRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        # spatial paddings
        latdist = (self.nlat + self.comm_size_h - 1) // self.comm_size_h
        self.latpad = latdist * self.comm_size_h - self.nlat
        londist = (self.nlon + self.comm_size_w - 1) // self.comm_size_w
        self.lonpad = londist * self.comm_size_w - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_h - 1) // self.comm_size_h
        self.lpad = ldist * self.comm_size_h - self.lmax
        mdist = (self.mmax + self.comm_size_w - 1) // self.comm_size_w
        self.mpad = mdist * self.comm_size_w - self.mmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        # we need to ensure that we can split the channels evenly
        assert x.shape[1] % self.comm_size_h == 0
        assert x.shape[1] % self.comm_size_w == 0

        # transpose: after that, channels are split, l is local:
        if self.comm_size_h > 1:
            xt = distributed_transpose_h.apply(x, (1, -2))
        else:
            xt = x

        # truncate
        xtt = xt[..., : self.lmax, :]

        # do first fft
        xf = torch.fft.ifft(xtt, n=self.nlat, dim=-2, norm="ortho")

        # transpose: after this, l is split and channels are local
        xfp = F.pad(xf, [0, 0, 0, self.latpad])

        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(xfp, (-2, 1))
        else:
            y = xfp

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            yt = distributed_transpose_w.apply(y, (1, -1))
        else:
            yt = y

        # truncate
        ytt = yt[..., : self.mmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(ytt, n=self.nlon, dim=-1, norm="ortho")

        # pad before we transpose back
        xp = F.pad(x, [0, self.lonpad])

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            out = distributed_transpose_w.apply(xp, (-1, 1))
        else:
            out = xp

        return out


# more complicated layers
class DistributedMLP(nn.Module):
    """Distributed MLP layer"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        output_bias=True,
        act_layer=nn.GELU,
        drop_rate=0.0,
        checkpointing=False,
    ):  # pragma: no cover

        super(DistributedMLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # get effective embedding size:
        comm_size = comm.get_size("matmul")
        assert (
            hidden_features % comm_size == 0
        ), "Error, hidden_features needs to be divisible by matmul_parallel_size"
        hidden_features_local = hidden_features // comm_size

        # first set of hp
        self.w1 = nn.Parameter(torch.ones(hidden_features_local, in_features, 1, 1))
        self.b1 = nn.Parameter(torch.zeros(hidden_features_local))

        # second set of hp
        self.w2 = nn.Parameter(torch.ones(out_features, hidden_features_local, 1, 1))

        if output_bias:
            self.b2 = nn.Parameter(torch.zeros(out_features))

        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop_rate > 0.0 else nn.Identity()

        # the weights are shared spatially
        self.w1.is_shared_mp = ["h", "w"]
        self.b1.is_shared_mp = ["h", "w"]
        self.w2.is_shared_mp = ["h", "w"]
        if output_bias:
            self.b2.is_shared_mp = [
                "matmul",
                "h",
                "w",
            ]  # this one is shared between all ranks

        # init weights
        self._init_weights()

    def _init_weights(self):  # pragma: no cover
        trunc_normal_(self.w1, std=0.02)
        nn.init.constant_(self.b1, 0.0)
        trunc_normal_(self.w2, std=0.02)
        if hasattr(self, "b2"):
            nn.init.constant_(self.b2, 0.0)

    def fwd(self, x):  # pragma: no cover
        """Forward function."""
        # we need to prepare paralellism here
        # spatial parallelism
        x = scatter_to_spatial_parallel_region(x, dim=-1)

        # prepare the matmul parallel part
        x = copy_to_matmul_parallel_region(x)

        # do the mlp
        x = F.conv2d(x, self.w1, bias=self.b1)
        x = self.act(x)
        x = self.drop(x)
        x = F.conv2d(x, self.w2, bias=None)
        x = reduce_from_matmul_parallel_region(x)
        if hasattr(self, "b2"):
            x = x + torch.reshape(self.b2, (1, -1, 1, 1))
        x = self.drop(x)

        # gather from spatial parallel region
        x = gather_from_spatial_parallel_region(x, dim=-1)

        return x

    @torch.jit.ignore
    def _checkpoint_forward(self, x):  # pragma: no cover
        return checkpoint(self.fwd, x)

    def forward(self, x):  # pragma: no cover
        if self.checkpointing:
            return self._checkpoint_forward(x)
        else:
            return self.fwd(x)


class DistributedPatchEmbed(nn.Module):
    """Distributed patch embedding layer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=True,
    ):  # pragma: no cover

        super(DistributedPatchEmbed, self).__init__()

        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        spatial_comm_size = comm.get_size("spatial")

        # compute parameters
        assert (
            img_size[1] // patch_size[1]
        ) % spatial_comm_size == 0, (
            "Error, make sure that the spatial comm size evenly divides patched W"
        )
        num_patches = ((img_size[1] // patch_size[1]) // spatial_comm_size) * (
            img_size[0] // patch_size[0]
        )
        self.img_size = (img_size[0], img_size[1] // spatial_comm_size)
        self.patch_size = patch_size
        self.num_patches = num_patches

        # get effective embedding size:
        if self.output_parallel:
            assert (
                embed_dim % matmul_comm_size == 0
            ), "Error, the embed_dim needs to be divisible by matmul_parallel_size"
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(
            in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size
        )

        # make sure we reduce them across rank
        self.proj.weight.is_shared_mp = ["h", "w"]
        self.proj.bias.is_shared_mp = ["h", "w"]

    def forward(self, x):  # pragma: no cover
        if self.input_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)

        if self.output_parallel:
            x = copy_to_matmul_parallel_region(x)

        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


@torch.jit.script
def compl_mul_add_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """complex multiplication and addition"""
    tmp = torch.einsum("bkixys,kiot->stbkoxy", a, b)
    res = (
        torch.stack(
            [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
        )
        + c
    )
    return res


@torch.jit.script
def compl_mul_add_fwd_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Performs a complex multiplication and addition operation on three tensors"""
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    tmp = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = tmp + cc
    return torch.view_as_real(res)


class DistributedAFNO2Dv2(nn.Module):
    """Distributed AFNO"""

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
        use_complex_kernels=False,
    ):  # pragma: no cover
        """Distributed AFNO2Dv2"""
        super(DistributedAFNO2Dv2, self).__init__()
        assert (
            hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        self.spatial_comm_size = comm.get_size("spatial")

        # select fft function handles
        if self.spatial_comm_size > 1:
            self.fft_handle = distributed_rfft2.apply
            self.ifft_handle = distributed_irfft2.apply
        else:
            self.fft_handle = torch.fft.rfft2
            self.ifft_handle = torch.fft.irfft2

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        assert (
            self.num_blocks % matmul_comm_size == 0
        ), "Error, num_blocks needs to be divisible by matmul_parallel_size"
        self.num_blocks_local = self.num_blocks // matmul_comm_size
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.mult_handle = (
            compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd
        )

        # model paralellism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # new
        # these weights need to be synced across all spatial ranks!
        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size,
                self.block_size * self.hidden_size_factor,
                2,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                1,
                1,
                2,
            )
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                self.block_size,
                2,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks_local, self.block_size, 1, 1, 2)
        )

        # make sure we reduce them across rank
        self.w1.is_shared_mp = ["h", "w"]
        self.b1.is_shared_mp = ["h", "w"]
        self.w2.is_shared_mp = ["h", "w"]
        self.b2.is_shared_mp = ["h", "w"]

    def forward(self, x):  # pragma: no cover
        if not self.input_is_matmul_parallel:
            # distribute data
            x = scatter_to_matmul_parallel_region(x, dim=1)

        # bias
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W_local = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        H_local = H // self.spatial_comm_size
        W = W_local * self.spatial_comm_size
        x = self.fft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.view(B, self.num_blocks_local, self.block_size, H_local, W // 2 + 1)

        # new
        x = torch.view_as_real(x)
        o2 = torch.zeros(x.shape, device=x.device)

        o1 = F.relu(
            self.mult_handle(
                x[
                    :,
                    :,
                    :,
                    total_modes - kept_modes : total_modes + kept_modes,
                    :kept_modes,
                    :,
                ],
                self.w1,
                self.b1,
            )
        )
        o2[
            :, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :
        ] = self.mult_handle(o1, self.w2, self.b2)

        # finalize
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H_local, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.type(dtype) + bias

        # gather
        if not self.output_is_matmul_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)

        return x


_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def config_logger(log_level=logging.INFO):  # pragma: no cover
    """
    Configure the logging basic settings with given log leve.
    """
    logging.basicConfig(format=_format, level=log_level)


def log_to_file(
    logger_name=None, log_level=logging.INFO, log_filename="tensorflow.log"
):  # pragma: no cover
    """
    Log to a file with the given log level.
    """
    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():  # pragma: no cover

    """
    Log the versions of git and torch.
    """
    import torch
    import subprocess

    logging.info("--------------- Versions ---------------")
    try:
        logging.info(
            "git branch: " + str(subprocess.check_output(["git", "branch"]).strip())
        )
        logging.info(
            "git hash: "
            + str(subprocess.check_output(["git", "rev-parse", "HEAD"]).strip())
        )
    except:
        pass
    logging.info("Torch: " + str(torch.__version__))
    logging.info("----------------------------------------")


class disable_logging(object):
    """
    A context manager to disable logging temporarily.
    """

    def __init__(self, level=logging.ERROR):
        """
        Initialize the context manager.
        """
        logging.disable(level=level)

    def __enter__(self):
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Exit the context manager and enable logging.
        """
        logging.disable(level=logging.NOTSET)







# dummy placeholders
_COMM_LIST = []
_COMM_NAMES = {}

# world comm
class comm:
    def get_size(comm_id: Union[str, int]) -> int:  # pragma: no cover
        """Returns the size of a specified communicator."""
        if isinstance(comm_id, int):
            cid = comm_id
        else:
            cid = _COMM_NAMES[comm_id] if (comm_id in _COMM_NAMES) else len(_COMM_LIST)

        if not dist.is_initialized() or (cid >= len(_COMM_LIST)):
            return 1
        else:
            return dist.get_world_size(group=_COMM_LIST[cid])


    def get_rank(comm_id: Union[str, int]) -> int:  # pragma: no cover
        """Returns the rank of a specified communicator."""
        if isinstance(comm_id, int):
            cid = comm_id
        else:
            cid = _COMM_NAMES[comm_id] if (comm_id in _COMM_NAMES) else len(_COMM_LIST)

        if not dist.is_initialized() or (cid >= len(_COMM_LIST)):
            return 0
        else:
            return dist.get_rank(group=_COMM_LIST[cid])


    def get_group(comm_id: Union[str, int]) -> int:  # pragma: no cover
        """Returns the group of a specified communicator."""
        if isinstance(comm_id, int):
            cid = comm_id
        else:
            cid = _COMM_NAMES[comm_id] if (comm_id in _COMM_NAMES) else len(_COMM_LIST)

        if not dist.is_initialized() or (cid >= len(_COMM_LIST)):
            raise IndexError(f"Error, comm with id {comm_id} not available.")
        else:
            return _COMM_LIST[cid]


    # specialized routines for world comms
    def get_world_size():  # pragma: no cover
        """Returns the world size"""
        if not dist.is_initialized():
            return 1
        else:
            return dist.get_world_size()


    def get_world_rank():  # pragma: no cover
        """Returns the world rank"""
        if not dist.is_initialized():
            return 0
        else:
            return dist.get_rank()


    def get_local_rank():  # pragma: no cover
        """Returns the local rank of the current process."""
        if os.getenv("LOCAL_RANK") is not None:
            # Use PyTorch env var if available
            return int(os.getenv("LOCAL_RANK"))

        if not dist.is_initialized():
            return 0
        else:
            return get_world_rank() % torch.cuda.device_count()


    def get_names():  # pragma: no cover
        """Returns the names of all available communicators."""
        return _COMM_NAMES


    def is_distributed(name: str):  # pragma: no cover
        """check if distributed."""
        return name in _COMM_NAMES


    # get
    def init(params, verbose=False):  # pragma: no cover
        """Initialize distributed training."""
        # set up global and local communicator
        if params.wireup_info == "env":
            world_size = int(os.getenv("WORLD_SIZE", 1))
            world_rank = int(os.getenv("RANK", 0))
            if os.getenv("WORLD_RANK") is not None:
                # Use WORLD_RANK if available for backwards compatibility
                world_rank = int(os.getenv("WORLD_RANK"))
            port = int(os.getenv("MASTER_PORT", 0))
            master_address = os.getenv("MASTER_ADDR")
            if os.getenv("MASTER_ADDRESS") is not None:
                # Use MASTER_ADDRESS if available for backwards compatibility
                master_address = int(os.getenv("MASTER_ADDRESS"))
        elif params.wireup_info == "mpi":
            import socket
            from mpi4py import MPI

            mpi_comm = MPI.COMM_WORLD.Dup()
            world_size = mpi_comm.Get_size()
            world_rank = mpi_comm.Get_rank()
            my_host = socket.gethostname()
            port = 29500
            master_address = None
            if world_rank == 0:
                master_address_info = socket.getaddrinfo(
                    my_host, port, family=socket.AF_INET, proto=socket.IPPROTO_TCP
                )
                master_address = master_address_info[0][-1][0]
            master_address = mpi_comm.bcast(master_address, root=0)
            os.environ["MASTER_ADDRESS"] = master_address
            os.environ["MASTER_PORT"] = str(port)
        else:
            raise ValueError(f"Error, wireup-info {params.wireup_info} not supported")
        # set local rank to 0 if env var not available
        local_rank = int(os.getenv("LOCAL_RANK", 0))

        if world_size > 1:
            with disable_logging():
                if params.wireup_store == "file":
                    wireup_file_path = os.getenv("WIREUP_FILE_PATH")
                    wireup_store = dist.FileStore(wireup_file_path, world_size)
                elif params.wireup_store == "tcp":
                    # create tcp store
                    wireup_store = dist.TCPStore(
                        host_name=master_address,
                        port=port,
                        world_size=world_size,
                        is_master=(world_rank == 0),
                        timeout=dt.timedelta(seconds=900),
                    )
                else:
                    wireup_store = None

                # initialize process groups
                dist.init_process_group(
                    backend="nccl",
                    rank=world_rank,
                    world_size=world_size,
                    store=wireup_store,
                )

                # get sizes
                world_size = get_world_size()
                world_rank = get_world_rank()
                local_rank = get_local_rank()

                # barrier
                dist.barrier(device_ids=[local_rank])

        # do individual wireup for model parallel comms:
        if hasattr(params, "model_parallel_sizes"):
            model_parallel_sizes = params.model_parallel_sizes
        else:
            model_parallel_sizes = [1]

        if hasattr(params, "model_parallel_names"):
            model_parallel_names = params.model_parallel_names
        else:
            model_parallel_names = ["model"]
        assert len(model_parallel_names) == len(
            model_parallel_sizes
        ), "Please specify names for your communicators"
        model_parallel_size = math.prod(model_parallel_sizes)
        params["model_parallel_size"] = model_parallel_size

        assert (
            world_size % model_parallel_size == 0
        ), "Error, please make sure that the product of model parallel ranks evenly divides the total number of ranks"

        # we set this to be orthogonal to the MP groups
        # we can play tricks with the ddp_group later, in case if all the weights are shared
        data_parallel_size = world_size // model_parallel_size

        # create orthogonal communicators first
        global _COMM_LIST
        global _COMM_NAMES
        if params.log_to_screen:
            logging.info("Starting Wireup")

        if world_size > 1:

            # set up the strides:
            model_parallel_sizes_reversed = model_parallel_sizes[::-1]
            model_grid = np.reshape(
                np.arange(0, model_parallel_size), model_parallel_sizes[::-1]
            )
            perm = np.roll(np.arange(0, len(model_parallel_sizes)), 1).tolist()
            ranks_lookup = {}

            comm_count = 0
            for mpname in model_parallel_names:
                base_group = np.reshape(model_grid, (-1, model_grid.shape[-1]))
                model_groups = []
                for goffset in range(0, world_size, model_parallel_size):
                    model_groups += sorted((goffset + base_group).tolist())

                if verbose and world_rank == 0:
                    print(f"Creating comm groups for id {mpname}: {model_groups}")

                for grp in model_groups:
                    if len(grp) > 1:
                        tmp_group = dist.new_group(ranks=grp)
                        if world_rank in grp:
                            _COMM_LIST.append(tmp_group)
                            _COMM_NAMES[mpname] = comm_count
                            comm_count += 1
                ranks_lookup[mpname] = model_groups

                # go for the next step
                model_grid = np.transpose(model_grid, perm)

            # now, we create a single communicator for h and w ranks
            if (get_size("h") == 1) and (get_size("w") > 1):
                if verbose and world_rank == 0:
                    print(f'Creating comm groups for id spatial: {ranks_lookup["w"]}')
                _COMM_LIST.append(get_group("w"))
                _COMM_NAMES["spatial"] = comm_count
                comm_count += 1
            elif (get_size("h") > 1) and (get_size("w") == 1):
                if verbose and world_rank == 0:
                    print(f'Creating comm groups for id spatial: {ranks_lookup["h"]}')
                _COMM_LIST.append(get_group("h"))
                _COMM_NAMES["spatial"] = comm_count
                comm_count += 1
            elif (get_size("h") > 1) and (get_size("w") > 1):
                # fuse the lists:
                def merge_ranks(list1, list2):
                    """Merge ranks"""
                    coll = list1 + list2
                    pooled = [set(subList) for subList in coll]
                    merging = True
                    while merging:
                        merging = False
                        for i, group in enumerate(pooled):
                            merged = next(
                                (g for g in pooled[i + 1 :] if g.intersection(group)), None
                            )
                            if not merged:
                                continue
                            group.update(merged)
                            pooled.remove(merged)
                            merging = True
                    return [list(x) for x in pooled]

                model_groups = merge_ranks(ranks_lookup["h"], ranks_lookup["w"])
                if verbose and world_rank == 0:
                    print(f"Creating comm groups for id spatial: {model_groups}")
                for grp in model_groups:
                    tmp_group = dist.new_group(ranks=grp)
                    if world_rank in grp:
                        _COMM_LIST.append(tmp_group)
                        _COMM_NAMES["spatial"] = comm_count
                        comm_count += 1

            # now the data and model comm:
            model_groups = np.reshape(
                np.arange(0, world_size), (-1, model_parallel_size)
            ).tolist()
            for grp in model_groups:
                if len(grp) > 1:
                    tmp_group = dist.new_group(ranks=grp)
                    if world_rank in grp:
                        _COMM_LIST.append(tmp_group)
                        _COMM_NAMES["model"] = comm_count
                        comm_count += 1

            if data_parallel_size == world_size:
                if verbose and world_rank == 0:
                    print(
                        f"Creating comm groups for id data: {[list(range(0, world_size))]}"
                    )

                _COMM_LIST.append(None)
                _COMM_NAMES["data"] = comm_count
            else:
                data_groups = [sorted(list(i)) for i in zip(*model_groups)]

                if verbose and world_rank == 0:
                    print(f"Creating comm groups for id data: {data_groups}")

                for grp in data_groups:
                    tmp_group = dist.new_group(ranks=grp)
                    if world_rank in grp:
                        _COMM_LIST.append(tmp_group)
                        _COMM_NAMES["data"] = comm_count

        # barrier
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

        if params.log_to_screen:
            logging.info("Finished Wireup")

        return


class DistributedInstanceNorm2d(nn.Module):
    """
    Computes a distributed instance norm using Welford's online algorithm
    """

    def __init__(
        self, num_features, eps=1e-05, affine=False, device=None, dtype=None
    ):  # pragma: no cover
        super(DistributedInstanceNorm2d, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
            self.weight.is_shared_mp = ["h", "w"]
            self.bias.is_shared_mp = ["h", "w"]

        self.gather_mode = "welford"

    @torch.jit.ignore
    def _gather_hw(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # gather the data over the spatial communicator
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")
        return xw

    @torch.jit.ignore
    def _gather_spatial(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # gather the data over the spatial communicator
        xs = gather_from_parallel_region(x, -1, "spatial")
        return xs

    def _stats_naive(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        """Computes the statistics in the naive way by first gathering the tensors and then computing them"""

        x = self._gather_hw(x)
        var, mean = torch.var_mean(x, dim=(-2, -1), unbiased=False, keepdim=True)

        return var, mean

    def _stats_welford(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        """Computes the statistics locally, then uses the Welford online algorithm to reduce them"""

        var, mean = torch.var_mean(x, dim=(-2, -1), unbiased=False, keepdim=False)
        # workaround to not use shapes, as otherwise cuda graphs won't work
        count = torch.ones_like(x[0, 0], requires_grad=False)
        count = torch.sum(count, dim=(-2, -1), keepdim=False)

        vars = self._gather_spatial(var.unsqueeze(-1))
        means = self._gather_spatial(mean.unsqueeze(-1))
        counts = self._gather_spatial(count.unsqueeze(-1))

        m2s = vars * counts

        mean = means[..., 0]
        m2 = m2s[..., 0]
        count = counts[..., 0]

        # use Welford's algorithm to accumulate them into a single mean and variance
        for i in range(1, comm.get_size("spatial")):
            delta = means[..., i] - mean
            m2 = (
                m2
                + m2s[..., i]
                + delta**2 * count * counts[..., i] / (count + counts[..., i])
            )
            if i == 1:
                mean = (mean * count + means[..., i] * counts[..., i]) / (
                    count + counts[..., i]
                )
            else:
                mean = mean + delta * counts[..., i] / (count + counts[..., i])

            # update the current count
            count = count + counts[..., i]

        var = m2 / count

        var = var.reshape(1, -1, 1, 1)
        mean = mean.reshape(1, -1, 1, 1)

        return var, mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        with amp.autocast(enabled=False):
            dtype = x.dtype
            x = x.float()

            # start by computing std and mean
            if self.gather_mode == "naive":
                var, mean = self._stats_naive(x)
            elif self.gather_mode == "welford":
                var, mean = self._stats_welford(x)
            else:
                raise ValueError(f"Unknown gather mode {self.gather_mode}")

            # this is absolutely necessary to get the correct graph in the backward pass
            mean = copy_to_spatial_parallel_region(mean)
            var = copy_to_spatial_parallel_region(var)

        x = x.to(dtype)
        mean = mean.to(dtype)
        var = var.to(dtype)

        # apply the normalization
        x = (x - mean) / torch.sqrt(var + self.eps)

        # affine transform if we use it
        if self.affine:
            x = self.weight * x + self.bias

        return x



class DistributedRealSHT(nn.Module):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        """
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = torch.from_numpy(pct)
        weights = torch.einsum('mlk,k->mlk', pct, weights)

        # split weights
        weights = split_tensor_along_dim(weights, dim=0, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth]

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[1]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * torch.pi * torch.fft.rfft(x, n=self.nlon, dim=-1, norm="forward")

        # truncate
        x = x[..., :self.mmax]

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.lat_shapes)

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # contraction
        xs = torch.einsum('...kmr,mlk->...lmr', x, self.weights.to(x.dtype)).contiguous()

        # cast to complex
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar	> 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        return x


class DistributedInverseRealSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute legende polynomials
        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        pct = torch.from_numpy(pct)

        # split in m
        pct = split_tensor_along_dim(pct, dim=0, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth]

        # register
        self.register_buffer('pct', pct, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[1]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.l_shapes)

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # einsum
        xs = torch.einsum('...lmr, mlk->...kmr', x, self.pct.to(x.dtype)).contiguous()
        #rl = torch.einsum('...lm, mlk->...km', x[..., 0], self.pct.to(x.dtype) )
        #im = torch.einsum('...lm, mlk->...km', x[..., 1], self.pct.to(x.dtype) )
        #xs = torch.stack((rl, im), -1).contiguous()

        # inverse FFT
        x = torch.view_as_complex(xs)

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        return x


class DistributedRealVectorSHT(nn.Module):
    """
    Defines a module for computing the forward (real) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last three dimensions of the input.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        """
        Initializes the vector SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: type of grid the data lives on
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # get the comms grid:
        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute weights
        weights = torch.from_numpy(w)
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        dpct = torch.from_numpy(dpct)

        # combine integration weights, normalization factor in to one:
        l = torch.arange(0, self.lmax)
        norm_factor = 1. / l / (l+1)
        norm_factor[0] = 1.
        weights = torch.einsum('dmlk,k,l->dmlk', dpct, weights, norm_factor)
        # since the second component is imaginary, we need to take complex conjugation into account
        weights[1] = -1 * weights[1]

        # we need to split in m, pad before:
        weights = split_tensor_along_dim(weights, dim=1, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth]

        # remember quadrature weights
        self.register_buffer('weights', weights, persistent=False)


    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(len(x.shape) >= 3)

        # we need to ensure that we can split the channels evenly
        num_chans = x.shape[1]

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.lon_shapes)

        # apply real fft in the longitudinal direction: make sure to truncate to nlon
        x = 2.0 * torch.pi * torch.fft.rfft(x, n=self.nlon, dim=-1, norm="forward")

        # truncate
        x = x[..., :self.mmax]

        # transpose: after this, m is split and c is local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        # transpose: after this, c is split and h is local
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.lat_shapes)

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # create output array
        xs = torch.zeros_like(x, dtype=x.dtype, device=x.device)

        # contraction - spheroidal component
        # real component
        xs[..., 0, :, :, 0] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 0], self.weights[0].to(xs.dtype)) \
                              - torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 1], self.weights[1].to(xs.dtype))
        # imag component
        xs[..., 0, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 1], self.weights[0].to(xs.dtype)) \
                              + torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 0], self.weights[1].to(xs.dtype))

        # contraction - toroidal component
        # real component
        xs[..., 1, :, :, 0] = - torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 1], self.weights[1].to(xs.dtype)) \
                              - torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 0], self.weights[0].to(xs.dtype))
        # imag component
        xs[..., 1, :, :, 1] =   torch.einsum('...km,mlk->...lm', x[..., 0, :, :, 0], self.weights[1].to(xs.dtype)) \
                              - torch.einsum('...km,mlk->...lm', x[..., 1, :, :, 1], self.weights[0].to(xs.dtype))

        # pad if required
        x = torch.view_as_complex(xs)

        # transpose: after this, l is split and c is local
        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        return x


class DistributedInverseRealVectorSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) vector SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """
    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        self.comm_size_polar = polar_group_size()
        self.comm_rank_polar = polar_group_rank()
        self.comm_size_azimuth = azimuth_group_size()
        self.comm_rank_azimuth = azimuth_group_rank()

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # compute splits
        self.lat_shapes = compute_split_shapes(self.nlat, self.comm_size_polar)
        self.lon_shapes = compute_split_shapes(self.nlon, self.comm_size_azimuth)
        self.l_shapes = compute_split_shapes(self.lmax, self.comm_size_polar)
        self.m_shapes = compute_split_shapes(self.mmax, self.comm_size_azimuth)

        # compute legende polynomials
        dpct = _precompute_dlegpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        dpct = torch.from_numpy(dpct)

        # split in m
        dpct = split_tensor_along_dim(dpct, dim=1, num_chunks=self.comm_size_azimuth)[self.comm_rank_azimuth]

        # register buffer
        self.register_buffer('dpct', dpct, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # store num channels
        num_chans = x.shape[1]

        # transpose: after that, channels are split, l is local:
        if self.comm_size_polar > 1:
            x = distributed_transpose_polar.apply(x, (1, -2), self.l_shapes)

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        # contraction - spheroidal component
        # real component
        srl =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[0].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[1].to(x.dtype))
        # imag component
        sim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[0].to(x.dtype)) \
              + torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[1].to(x.dtype))

        # contraction - toroidal component
        # real component
        trl = - torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 1], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 0], self.dpct[0].to(x.dtype))
        # imag component
        tim =   torch.einsum('...lm,mlk->...km', x[..., 0, :, :, 0], self.dpct[1].to(x.dtype)) \
              - torch.einsum('...lm,mlk->...km', x[..., 1, :, :, 1], self.dpct[0].to(x.dtype))

        # reassemble
        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)

        # convert to complex
        x = torch.view_as_complex(xs)

        if self.comm_size_polar > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_polar)
            x = distributed_transpose_polar.apply(x, (-2, 1), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_azimuth > 1:
            x = distributed_transpose_azimuth.apply(x, (1, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        # transpose: after this, m is split and channels are local
        if self.comm_size_azimuth > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_azimuth)
            x = distributed_transpose_azimuth.apply(x, (-1, 1), chan_shapes)

        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "SFNO"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class SpectralFilterLayer(nn.Module):
    """Spectral filter layer"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="block-diagonal",
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        hidden_size_factor=1,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        drop_rate=0.0,
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, DistributedRealSHT)
        ):
            self.filter = SpectralAttentionS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                operator_type=operator_type,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        elif filter_type == "non-linear" and (
            isinstance(forward_transform, RealFFT2)
            or isinstance(forward_transform, DistributedRealFFT2)
        ):
            self.filter = SpectralAttention2d(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        # spectral transform is passed to the module
        elif filter_type == "linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, DistributedRealSHT)
        ):
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=True,
                use_tensorly=False if factorization is None else True,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator Block"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=(nn.LayerNorm, nn.LayerNorm),
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        use_mlp=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=0,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
            self.input_shape_loc = (
                forward_transform.nlat_local,
                forward_transform.nlon_local,
            )
            self.output_shape_loc = (
                inverse_transform.nlat_local,
                inverse_transform.nlon_local,
            )
        else:
            self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
            self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = norm_layer[0]()

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            sparsity_threshold,
            use_complex_kernels=use_complex_kernels,
            hidden_size_factor=mlp_ratio,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            drop_rate=drop_rate,
        )

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

        if filter_type == "linear" or filter_type == "real linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()

        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

    def forward(self, x):

        x_norm = torch.zeros_like(x)
        x_norm[..., : self.input_shape_loc[0], : self.input_shape_loc[1]] = self.norm0(
            x[..., : self.input_shape_loc[0], : self.input_shape_loc[1]]
        )
        x, residual = self.filter(x_norm)

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x_norm = torch.zeros_like(x)
        x_norm[
            ..., : self.output_shape_loc[0], : self.output_shape_loc[1]
        ] = self.norm1(x[..., : self.output_shape_loc[0], : self.output_shape_loc[1]])
        x = x_norm

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(Module):

    def __init__(
        self,
        params: dict,
        spectral_transform: str = "sht",
        filter_type: str = "non-linear",
        operator_type: str = "diagonal",
        img_shape: Tuple[int] = (721, 1440),
        scale_factor: int = 16,
        in_chans: int = 2,
        out_chans: int = 2,
        embed_dim: int = 256,
        num_layers: int = 12,
        use_mlp: int = True,
        mlp_ratio: int = 2.0,
        activation_function: str = "gelu",
        encoder_layers: int = 1,
        pos_embed: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.0,
        normalization_layer: str = "instance_norm",
        hard_thresholding_fraction: float = 1.0,
        use_complex_kernels: bool = True,
        big_skip: bool = True,
        rank: float = 1.0,
        factorization: Any = None,
        separable: bool = False,
        complex_network: bool = True,
        complex_activation: str = "real",
        spectral_layers: int = 3,
        checkpointing: int = 0,
    ):

        super(SphericalFourierNeuralOperatorNet, self).__init__(meta=MetaData())

        self.params = params
        self.spectral_transform = (
            params.spectral_transform
            if hasattr(params, "spectral_transform")
            else spectral_transform
        )
        self.filter_type = (
            params.filter_type if hasattr(params, "filter_type") else filter_type
        )
        self.operator_type = (
            params.operator_type if hasattr(params, "operator_type") else operator_type
        )
        self.img_shape = (
            (params.img_shape_x, params.img_shape_y)
            if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y")
            else img_shape
        )
        self.scale_factor = (
            params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        )
        self.in_chans = (
            params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        )
        self.out_chans = (
            params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        )
        self.embed_dim = self.num_features = (
            params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        )
        self.num_layers = (
            params.num_layers if hasattr(params, "num_layers") else num_layers
        )
        self.num_blocks = (
            params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        )
        self.hard_thresholding_fraction = (
            params.hard_thresholding_fraction
            if hasattr(params, "hard_thresholding_fraction")
            else hard_thresholding_fraction
        )
        self.normalization_layer = (
            params.normalization_layer
            if hasattr(params, "normalization_layer")
            else normalization_layer
        )
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp
        self.activation_function = (
            params.activation_function
            if hasattr(params, "activation_function")
            else activation_function
        )
        self.encoder_layers = (
            params.encoder_layers
            if hasattr(params, "encoder_layers")
            else encoder_layers
        )
        self.pos_embed = params.pos_embed if hasattr(params, "pos_embed") else pos_embed
        self.big_skip = params.big_skip if hasattr(params, "big_skip") else big_skip
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.factorization = (
            params.factorization if hasattr(params, "factorization") else factorization
        )
        self.separable = params.separable if hasattr(params, "separable") else separable
        self.complex_network = (
            params.complex_network
            if hasattr(params, "complex_network")
            else complex_network
        )
        self.complex_activation = (
            params.complex_activation
            if hasattr(params, "complex_activation")
            else complex_activation
        )
        self.spectral_layers = (
            params.spectral_layers
            if hasattr(params, "spectral_layers")
            else spectral_layers
        )
        self.checkpointing = (
            params.checkpointing if hasattr(params, "checkpointing") else checkpointing
        )
        data_grid = params.data_grid if hasattr(params, "data_grid") else "legendre-gauss"
        # self.pretrain_encoding = params.pretrain_encoding if hasattr(params, "pretrain_encoding") else False

        # compute the downscaled image size
        self.h = int(self.img_shape[0] // self.scale_factor)
        self.w = int(self.img_shape[1] // self.scale_factor)

        # Compute the maximum frequencies in h and in w
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

        # determine the global padding
        img_dist_h = (self.img_shape[0] + comm.get_size("h") - 1) // comm.get_size("h")
        img_dist_w = (self.img_shape[1] + comm.get_size("w") - 1) // comm.get_size("w")
        self.padding = (
            img_dist_h * comm.get_size("h") - self.img_shape[0],
            img_dist_w * comm.get_size("w") - self.img_shape[1],
        )

        # prepare the spectral transforms
        if self.spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # parallelism
            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = (
                    None if (comm.get_size("w") == 1) else comm.get_group("w")
                )
                thd.init(polar_group, azimuth_group)
                sht_handle = DistributedRealSHT
                isht_handle = DistributedInverseRealSHT

            # set up
            self.trans_down = sht_handle(
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
            ).float()
            self.itrans_up = isht_handle(
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
            ).float()
            self.trans = sht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"  # was legendre-gauss
            ).float()
            self.itrans = isht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"  # was legendre-gauss
            ).float()

        elif self.spectral_transform == "fft":
            fft_handle = th.RealFFT2
            ifft_handle = th.InverseRealFFT2

            # effective image size:
            self.img_shape_eff = [
                self.img_shape[0] + self.padding[0],
                self.img_shape[1] + self.padding[1],
            ]
            self.img_shape_loc = [
                self.img_shape_eff[0] // comm.get_size("h"),
                self.img_shape_eff[1] // comm.get_size("w"),
            ]

            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2

            self.trans_down = fft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans_up = ifft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.trans = fft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans = ifft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
            self.img_shape_loc = (
                self.trans_down.nlat_local,
                self.trans_down.nlon_local,
            )
            self.img_shape_eff = [
                self.trans_down.nlat_local + self.trans_down.nlatpad_local,
                self.trans_down.nlon_local + self.trans_down.nlonpad_local,
            ]
            self.h_loc = self.itrans.nlat_local
            self.w_loc = self.itrans.nlon_local
        else:
            self.img_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
            self.img_shape_eff = [self.trans_down.nlat, self.trans_down.nlon]
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

        # determine activation function
        if self.activation_function == "relu":
            self.activation_function = nn.ReLU
        elif self.activation_function == "gelu":
            self.activation_function = nn.GELU
        elif self.activation_function == "silu":
            self.activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {self.activation_function}")

        # encoder
        encoder_hidden_dim = self.embed_dim
        current_dim = self.in_chans
        encoder_modules = []
        for i in range(self.encoder_layers):
            encoder_modules.append(
                nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            )
            encoder_modules.append(self.activation_function())
            current_dim = encoder_hidden_dim
        encoder_modules.append(nn.Conv2d(current_dim, self.embed_dim, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(
                nn.LayerNorm,
                normalized_shape=(self.img_shape_loc[0], self.img_shape_loc[1]),
                eps=1e-6,
            )
            norm_layer1 = partial(
                nn.LayerNorm, normalized_shape=(self.h_loc, self.w_loc), eps=1e-6
            )
        elif self.normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer0 = partial(
                    DistributedInstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                )
            else:
                norm_layer0 = partial(
                    nn.InstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                    track_running_stats=False,
                )
            norm_layer1 = norm_layer0
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(
                f"Error, normalization {self.normalization_layer} not implemented."
            )

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear"
            outer_skip = "identity"

            if first_layer:
                norm_layer = (norm_layer0, norm_layer1)
            elif last_layer:
                norm_layer = (norm_layer1, norm_layer0)
            else:
                norm_layer = (norm_layer1, norm_layer1)

            filter_type = self.filter_type

            operator_type = self.operator_type

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                sparsity_threshold=sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
                rank=self.rank,
                factorization=self.factorization,
                separable=self.separable,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                checkpointing=self.checkpointing,
            )

            self.blocks.append(block)

        # decoder
        decoder_hidden_dim = self.embed_dim
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_modules = []
        for i in range(self.encoder_layers):
            decoder_modules.append(
                nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            )
            decoder_modules.append(self.activation_function())
            current_dim = decoder_hidden_dim
        decoder_modules.append(nn.Conv2d(current_dim, self.out_chans, 1, bias=False))
        self.decoder = nn.Sequential(*decoder_modules)

        # learned position embedding
        if self.pos_embed:
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, self.embed_dim, self.img_shape_loc[0], self.img_shape_loc[1]
                )
            )
            # self.pos_embed = nn.Parameter( torch.zeros(1, self.embed_dim, self.img_shape_eff[0], self.img_shape_eff[1]) )
            self.pos_embed.is_shared_mp = ["matmul"]
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Helper routine for weight initialization"""
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):  # pragma: no cover
        """Helper"""
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):

        for blk in self.blocks:
            if self.checkpointing >= 3:
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        return x

    def forward(self, x):

        # save big skip
        if self.big_skip:
            residual = x

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):

            # old way of treating unequally shaped weights
            if self.img_shape_loc != self.img_shape_eff:
                xp = torch.zeros_like(x)
                xp[..., : self.img_shape_loc[0], : self.img_shape_loc[1]] = (
                    x[..., : self.img_shape_loc[0], : self.img_shape_loc[1]]
                    + self.pos_embed
                )
                x = xp
            else:
                x = x + self.pos_embed

        # maybe clean the padding just in case

        x = self.pos_drop(x)

        x = self._forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x)
        else:
            x = self.decoder(x)

        return x



import numpy as np
import torch
import torch.nn as nn
import torch.fft



class RealSHT(nn.Module):
    """
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        """
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        weights = torch.einsum('mlk,k->mlk', pct, weights)

        # remember quadrature weights
        # if USE_FIX:
        self.weights = weights.float()
        # else:
        #     self.register_buffer('weights', weights, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.nlat)
        assert(x.shape[-1] == self.nlon)

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)

        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

        # contraction
        if USE_FIX:
            self.weights = self.weights.to(x.device)
        xout[..., 0] = torch.einsum('...km,mlk->...lm', x[..., :self.mmax, 0], self.weights)
        xout[..., 1] = torch.einsum('...km,mlk->...lm', x[..., :self.mmax, 1], self.weights)
        x = torch.view_as_complex(xout)

        return x

class InverseRealSHT(nn.Module):
    """
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions
        self.mmax = mmax or self.nlon // 2 + 1

        pct = precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)

        # register buffer
        if USE_FIX:
            self.pct = pct.float()
        else:
            self.register_buffer('pct', pct, persistent=False)

    def extra_repr(self):
        """
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.lmax)
        assert(x.shape[-1] == self.mmax)

        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)

        if USE_FIX:
            self.pct = self.pct.to(x.device)
        rl = torch.einsum('...lm, mlk->...km', x[..., 0], self.pct )
        im = torch.einsum('...lm, mlk->...km', x[..., 1], self.pct )
        xs = torch.stack((rl, im), -1)

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm="forward")

        return x



class SpectralLoss(nn.Module):
    """Spectral loss"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = True,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
    ):  # pragma: no cover
        super(SpectralLoss, self).__init__()

        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared

        self.sht = th.RealSHT(*img_size, grid="legendre-gauss").float()
        spectral_weights = torch.arange(self.sht.lmax).float()
        spectral_weights = spectral_weights + 1
        self.register_buffer("spectral_weights", spectral_weights)

    def abs(self, prd: torch.Tensor, tar: torch.Tensor):  # pragma: no cover
        """Computes the absolute loss"""
        num_examples = prd.size()[0]

        # compute coefficients
        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        norm2 = (self.spectral_weights * norm2).reshape(num_examples, -1).sum(dim=-1)

        if not self.squared:
            norm2 = torch.sqrt(norm2)

        all_norms = norm2

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(
        self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):  # pragma: no cover
        """Computes the relative loss"""
        num_examples = prd.size()[0]

        # compute coefficients
        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        norm2 = (self.spectral_weights * norm2).reshape(num_examples, -1).sum(dim=-1)

        # compute coefficients
        tar_coeffs = torch.view_as_real(self.sht(tar))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(
            tar_coeffs[..., :, 1:], dim=-1
        )
        tar_norm2 = (
            (self.spectral_weights * tar_norm2).reshape(num_examples, -1).sum(dim=-1)
        )

        retval = tar_norm2 / norm2

        if mask is not None:
            retval = retval * mask

        if self.reduction:
            if self.size_average:
                if mask is None:
                    retval = torch.mean(retval)
                else:
                    retval = torch.sum(retval) / torch.sum(mask)
            else:
                retval = torch.sum(retval)

        if not self.squared:
            retval

        return retval

    def forward(
        self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):  # pragma: no cover
        if self.absolute:
            loss = self.abs(prd, tar)
        else:
            loss = self.rel(prd, tar, mask)

        return loss







