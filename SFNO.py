from functools import partial
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Any, Tuple
import logging

from typing import Union
from pathlib import Path

import torch.nn.functional as F
import torch.fft
from torch.cuda import amp
import math
from tqdm import tqdm

from utils import *
from contraction import *
from factorization import *
# from torch_harmonics import *


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





class SphericalFourierNeuralOperatorNet(nn.Module):

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
            trunc_normal_(m.weight, std=0.1)  # was 0.02
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
