# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.utils.checkpoint import checkpoint

from torch_harmonics.quadrature import legendre_gauss_weights, lobatto_weights, clenshaw_curtiss_weights
from torch_harmonics.legendre import _precompute_legpoly, _precompute_dlegpoly
from torch_harmonics.distributed import polar_group_size, azimuth_group_size, distributed_transpose_azimuth, distributed_transpose_polar
from torch_harmonics.distributed import polar_group_rank, azimuth_group_rank
from torch_harmonics.distributed import compute_split_shapes, split_tensor_along_dim

# numpy packages
import numpy as np

# import torch_harmonics
from torch_harmonics import *
import torch_harmonics as th
import torch_harmonics.distributed as thd
import torch.distributed as dist

# misc
from functools import partial
from dataclasses import dataclass
from typing import Any, Tuple
import math
import warnings
import os
import logging
import datetime as dt
from typing import Union
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path


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

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# jitted PyTorch contractions:
def _contract_dense_pytorch(
    x, weight, separable=False, operator_type="diagonal"
):  # pragma: no cover

    # to cheat the fused optimizers convert to real here
    x = torch.view_as_real(x)

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

    if implementation == "reconstructed":
        return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense_pytorch
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

        if isinstance(self.inverse_transform, InverseRealSHT):
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

        # spectral transform is passed to the module
        if filter_type == "linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, RealSHT)
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
            MLPH = MLP
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
        filter_type: str = "linear",
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
            sht_handle = RealSHT
            isht_handle = InverseRealSHT

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
        weights = torch.from_numpy(w).to(device)
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = torch.from_numpy(pct).to(device)
        weights = torch.einsum('mlk,k->mlk', pct, weights)

        # remember quadrature weights
        # if USE_FIX:
        self.weights = weights.float().to(device)
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

        self.sht = RealSHT(*img_size, grid="legendre-gauss").float()
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



import time

def spectral_regularizer(prd, tar, relative=False, squared=True):
    # compute coefficients
    diff = (prd-tar)

    shtdiff = sht(diff)
    coeffs = torch.view_as_real(shtdiff)

    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2    # take the real part only

    norm2 = 2 * torch.sum(coeffs[..., :, 24:], dim=-1)  # regularize the wave number beyond 5
    loss_reg = torch.sum(norm2, dim=(-1,-2))

    if not squared:
        loss_reg = torch.sqrt(loss_reg)
    loss_reg = loss_reg.mean()

    return loss_reg

num_elements = 48

# Generate evenly spaced values between -π/2 and π/2
cos_weight_reg = torch.cos(torch.linspace(-np.pi, np.pi, num_elements)).to(device).reshape(1,48) + 1


mse_loss = nn.MSELoss()
def train_model(model, train_set, test_set, optimizer, scheduler=None, nepochs=20, nfuture=0, num_examples=256, num_valid=8, loss_fn='l2', reg_rate=0):

    train_start = time.time()

    train_loader_1 = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader_1 = DataLoader(test_set, batch_size=16, shuffle=True)

    infer_bias = 1e+20
    recall_count = 0
    for epoch in range(nepochs):
        print(epoch)
        if epoch >= 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
        if recall_count > 20:
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

            lat_index = np.r_[7:15, 32:40]
            #lat_index = np.r_[0:48]
            out_fft = torch.mean(torch.abs(torch.fft.rfft(prd[:,:,lat_index,:],dim=3)),dim=2)
            target_fft = torch.mean(torch.abs(torch.fft.rfft(tar[:,:,lat_index,:],dim=3)),dim=2)

            quad_weight_reg = quad_weights.reshape(1,48)

            wave_index = np.r_[0:48]
            #wave_index = np.r_[7:15,32:40]
            loss_fft = torch.abs(out_fft[...,wave_index] - target_fft[...,wave_index])
            loss_reg = torch.mean(loss_fft,dim)
            
            loss.backward()
            optimizer.step()
            prd_plt = prd
            tar_plt = tar
            
        with torch.no_grad():
            for inp, tar in test_loader_1:
                inp = inp.to(device)
                tar = tar.to(device)
                prd = model(inp)
                if loss_fn == 'l2':
                    loss = l2loss_sphere(prd, tar, relative=True).mean()
                elif loss_fn == "spectral l2":
                    loss = spectral_l2loss_sphere(prd, tar, relative=True)
                prd = prd.to(device)

                valid_loss += loss.item() * inp.size(0)
                prd_testplt = prd
                tar_testplt = tar
        valid_loss = valid_loss / len(test_loader_1.dataset)
        epoch_time = time.time() - epoch_start

        if scheduler is not None:
            scheduler.step()

        if (epoch+1) % 5 == 0:
            print(loss, valid_loss)
            with torch.no_grad():
                pred_frame = input_dataset[0].reshape(1,7,48,96) # T, SH, U, V, SP, TISR, ORO
                pred_frame = pred_frame.to(device)
                temp_bias = torch.zeros(48,96).to(device)
                for k in range(7500):
                    previous_frame = pred_frame[:,:5,:,:]
                    pred_frame = model(pred_frame) # 6
                    temp_bias += pred_frame[0,1,:,:].clone().detach()
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
                print(infer_bias)
                if epoch > 100:
                    print(infer_bias)
                    if temp_bias <= infer_bias:
                        infer_bias = temp_bias
                        torch.save(model.state_dict(), 'regular_training_checkpoint.pth')
                    else:
                        state_pth = torch.load('regular_training_checkpoint.pth')
                        model.load_state_dict(state_pth)
                        recall_count += 1


