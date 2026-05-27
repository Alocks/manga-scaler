import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort  # Used for direct runtime optimization
from pathlib import Path
from onnxconverter_common import float16  # type: ignore[reportMissingImports]
from realesrgan import RRDBNet

try:
    from onnxsim import simplify  # type: ignore[reportMissingImports]
except ImportError:
    simplify = None


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style network used by realesr-general-x4v3."""

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=4,
        act_type='prelu',
    ):
        super().__init__()
        self.upscale = upscale

        def make_act():
            if act_type == 'relu':
                return nn.ReLU(inplace=True)
            if act_type == 'leakyrelu':
                return nn.LeakyReLU(negative_slope=0.1, inplace=True)
            if act_type == 'prelu':
                return nn.PReLU(num_parameters=num_feat)
            raise ValueError(f"Unsupported activation: {act_type}")

        layers = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), make_act()]
        for _ in range(num_conv):
            layers.extend([nn.Conv2d(num_feat, num_feat, 3, 1, 1), make_act()])
        layers.append(
            nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)
        )

        self.body = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = self.upsampler(self.body(x))
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out + base


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            1,
            1,
            0,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            in_channels // reduction,
            in_channels,
            1,
            1,
            0,
            bias=bias,
        )

    def forward(self, x):
        x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = F.relu(self.conv1(x0), inplace=True)
        x0 = torch.sigmoid(self.conv2(x0))
        return x * x0


class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.seblock = (
            SEBlock(out_channels, reduction=8, bias=True) if se else None
        )

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z


class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        return self.conv_bottom(x3)


class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)
        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

    def forward(self, x, alpha=1):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3) * alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        return self.conv_bottom(x5)


class UpCunet2x(nn.Module):
    """Real-CUGAN UpCunet2x — fully convolutional, supports dynamic spatial input sizes.

    The architecture is a cascade of UNet1 (2x deconv upsampler) and UNet2
    (residual refinement).  All operations are purely convolutional with
    symmetric padding/crop pairs that cancel out for any even-sized input,
    so no fixed tile size is required.
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x):
        # 18-pixel reflect pad provides border context for the first conv layer.
        # The symmetric -20 crop on unet1's output exactly cancels the padding
        # contribution, so the output is always 2× the input for any even H/W.
        x = F.pad(x, (18, 18, 18, 18), mode='reflect')
        x1 = self.unet1(x)
        x2 = self.unet2(x1, alpha=1)
        x1 = F.pad(x1, (-20, -20, -20, -20))
        # Clamp to [0, 1]: unet1 + unet2 is a two-path sum and can naturally
        # exceed 1.0 for bright pixels; the browser runtime expects [0, 1] output.
        return torch.clamp(torch.add(x2, x1), 0.0, 1.0)


class UpCunet2xUNet1Only(nn.Module):
    """Fast Real-CUGAN variant that runs only UNet1 (no UNet2 refinement)."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)

    def forward(self, x):
        x = F.pad(x, (18, 18, 18, 18), mode='reflect')
        x1 = self.unet1(x)
        x1 = F.pad(x1, (-20, -20, -20, -20))
        return torch.clamp(x1, 0.0, 1.0)


# ======================================================
# MoSR GPS ARCHITECTURE
# ======================================================

class _MoSRLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _GPSUpsampler(nn.Module):
    """Geo-ensemble Pixel Shuffle upsampler."""

    def __init__(self, dim: int, scale: int, out_ch: int = 3, kernel_size: int = 3):
        super().__init__()
        self.in_to_k = nn.Conv2d(dim, scale * scale * out_ch * 8, kernel_size, 1, kernel_size // 2)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_to_k(x)
        x = x.reshape(x.shape[0], 8, -1, x.shape[-2], x.shape[-1])
        x = x.mean(dim=1)
        return self.ps(x)


class _MoSRConvBlock(nn.Module):
    """Two-conv residual block used as the MoSR shortcut branch."""

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.Mish(),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.conv11(x)


class _GatedCNNBlock(nn.Module):
    """Modernized MambaOut block used inside MoSR gblocks."""

    def __init__(self, dim: int, expansion_ratio: float = 1.5, conv_ratio: float = 1.0, kernel_size: int = 7):
        super().__init__()
        hidden = int(expansion_ratio * dim)
        conv_channels = int(conv_ratio * dim)
        self.split_indices = [hidden, hidden - conv_channels, conv_channels]
        self.norm = _MoSRLayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 3, 1, 1)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size, 1, kernel_size // 2, groups=conv_channels)
        self.fc2 = nn.Conv2d(hidden, dim, 3, 1, 1)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=1)
        c = self.conv(c)
        x = self.act(self.fc2(self.act(g) * torch.cat((i, c), dim=1)))
        return x + (shortcut - 0.5)


class MoSRGPS(nn.Module):
    """MoSR with GPS (Geo-ensemble Pixel Shuffle) upsampler."""

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_block: int = 24,
        upscale: int = 2,
        kernel_size: int = 7,
        expansion_ratio: float = 1.5,
        conv_ratio: float = 1.0,
    ):
        super().__init__()
        self.gblocks = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [_GatedCNNBlock(dim, expansion_ratio, conv_ratio, kernel_size) for _ in range(n_block)]
            + [
                nn.Conv2d(dim, dim * 2, 3, 1, 1),
                nn.Mish(),
                nn.Conv2d(dim * 2, dim, 3, 1, 1),
                nn.Mish(),
                nn.Conv2d(dim, dim, 1, 1),
            ]
        )
        self.shortcut = _MoSRConvBlock(in_ch, dim)
        self.upsampler = _GPSUpsampler(dim, upscale, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampler(self.gblocks(x) + (self.shortcut(x) - 0.5))


# ======================================================
# REAL-PLKSR ARCHITECTURE
# ======================================================

class _DCCM(nn.Sequential):
    """Doubled Convolutional Channel Mixer."""

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )


class _PLKConv2d(nn.Module):
    """Partial Large Kernel Convolutional Layer (processes only first `dim` channels)."""

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        self.idx = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always use the non-in-place split path for ONNX compatibility.
        x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
        return torch.cat([self.conv(x1), x2], dim=1)


class _EA(nn.Module):
    """Element-wise Attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)


class _PLKBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, split_ratio: float, norm_groups: int, use_ea: bool = True):
        super().__init__()
        pdim = int(dim * split_ratio)
        self.channel_mixer = _DCCM(dim)
        self.lk = _PLKConv2d(pdim, kernel_size)
        self.attn = _EA(dim) if use_ea else nn.Identity()
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        self.norm = nn.GroupNorm(norm_groups, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)
        return x + x_skip


class RealPLKSR(nn.Module):
    """Real-PLKSR: Partial Large Kernel CNNs for Efficient Super-Resolution."""

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = 2,
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        use_ea: bool = True,
        norm_groups: int = 4,
    ):
        super().__init__()
        self.upscale = upscaling_factor
        self.feats = nn.Sequential(
            *[nn.Conv2d(in_ch, dim, 3, 1, 1)]
            + [_PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea) for _ in range(n_blocks)]
            + [nn.Dropout2d(0)]
            + [nn.Conv2d(dim, out_ch * upscaling_factor ** 2, 3, 1, 1)]
        )
        self.to_img = nn.PixelShuffle(upscaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_img(
            self.feats(x) + torch.repeat_interleave(x, repeats=self.upscale ** 2, dim=1)
        )


def _remap_old_esrgan_keys(old_sd):
    """Convert old-arch ESRGAN state dict keys to BasicSR RRDBNet-compatible names.

    Old-arch layout:
      model.0                        -> conv_first
      model.1.sub.N.RDBN.convK.0     -> body.N.rdbN.convK   (N = 0..num_block-1)
      model.1.sub.<last>             -> conv_body            (trunk Conv2d, last sub element)
      model.3                        -> conv_up1
      model.6                        -> conv_up2
      model.8                        -> conv_hr
      model.10                       -> conv_last
    """
    mapping = {
        'model.0.': 'conv_first.',
        'model.3.': 'conv_up1.',
        'model.6.': 'conv_up2.',
        'model.8.': 'conv_hr.',
        'model.10.': 'conv_last.',
    }
    new_sd = {}
    for k, v in old_sd.items():
        if k.startswith('model.1.sub.'):
            parts = k.split('.')
            if len(parts) == 5:
                # model.1.sub.<last>.{weight|bias} — trunk Conv2d = conv_body
                new_k = f'conv_body.{parts[4]}'
            else:
                # model.1.sub.N.RDBN.convK.0.{weight|bias}
                block_idx = parts[3]
                rdb_name = parts[4].lower()   # RDB1 -> rdb1
                conv_name = parts[5]          # conv1 .. conv5
                param_name = parts[7]         # weight or bias
                new_k = f'body.{block_idx}.{rdb_name}.{conv_name}.{param_name}'
        else:
            new_k = k
            for old_prefix, new_prefix in mapping.items():
                if k.startswith(old_prefix):
                    new_k = new_prefix + k[len(old_prefix):]
                    break
        new_sd[new_k] = v
    return new_sd


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'params_ema' in checkpoint:
            return checkpoint['params_ema']
        if 'params' in checkpoint:
            return checkpoint['params']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


class _ClampedModel(nn.Module):
    """Wraps any model and clamps its output to [0, 1].

    Inserting an explicit Clip op into the ONNX graph guarantees that the
    exported model is self-contained with respect to output range, regardless
    of the underlying architecture (residual sums, no final activation, etc.).
    The browser runtime then never needs to heuristically detect the value scale.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.model(x), 0.0, 1.0)


class _ScaledOutputModel(nn.Module):
    """Wraps a model and rescales its output spatial size by a fixed factor."""

    def __init__(self, model: nn.Module, scale_factor: float) -> None:
        super().__init__()
        self.model = model
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return F.interpolate(
            y,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=False,
        )


def _export_dynamic_onnx(model, onnx_output_path, sample_size=512):
    """Exports model with dynamic batch/height/width axes.

    Use for fully-convolutional models (ESRGAN, MoSR, RPLKSR) that support
    arbitrary spatial input sizes.  The resulting ONNX model accepts any tile
    dimension, removing the 512x512 hard constraint from the static export.

    The classic (non-dynamo) exporter is used here because torch.onnx.export
    with dynamo=True requires a different API for dynamic shapes (dynamic_shapes
    + torch.export.Dim).  The classic exporter and dynamic_axes dict are
    well-tested and produce correct dynamic ONNX graphs at opset 18.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, sample_size, sample_size, requires_grad=False)
    print(f"Exporting dynamic ONNX model to {onnx_output_path}...")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            dynamo=False,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input':  {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 2: 'height', 3: 'width'},
            },
        )


def _export_static_onnx(model, onnx_output_path, size=512):
    """Exports clean static graphs tailored specifically for stable WebGL/WebGPU runtimes."""
    dummy_input = torch.randn(1, 3, size, size, requires_grad=False)
    print(f"Compiling computational graph into {onnx_output_path}...")
    
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            dynamo=True,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # Dynamic axes omitted purposefully to guarantee static pipeline execution profiles.
        )


def _postprocess_onnx_for_web(onnx_output_path):
    print(f"Applying web-focused optimizations to {onnx_output_path}...")
    model = onnx.load(onnx_output_path)
    model = onnx.shape_inference.infer_shapes(model)

    # 1. Graph simplification MUST be processed while tensors match float32 precision
    if simplify is not None:
        simplified_model, check = simplify(model)
        if check:
            model = simplified_model
        else:
            print("ONNX simplifier check failed; maintaining fallback structure.")
    else:
        print("onnxsim module unavailable; skipping simplification pass.")
    
    # 2. Safely downcast the model logic to FP16 while preserving outermost IO signatures
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,          # Essential for continuous compatibility with JS Float32Arrays
        disable_shape_infer=False,
        op_block_list=[
            'Resize', 'Upsample',    # Upsampling — precision-sensitive interpolation
            'Add',                   # Residual/skip connections — values accumulate across blocks
            'Mul',                   # SE-block channel scaling — multiplicative blow-up in FP16
            'Sigmoid',               # Gates in SE blocks — FP16 saturates near boundaries
        ],
    )
    onnx.checker.check_model(model)
    onnx.save(model, onnx_output_path)

    # 3. Apply conservative ORT optimisations.
    #    ORT_ENABLE_BASIC is intentionally chosen over ORT_ENABLE_EXTENDED.
    #    EXTENDED performs Cast-elimination passes that silently remove the
    #    float32→float16 input Cast nodes added by keep_io_types=True, causing
    #    the exported model to require float16 tensors at inference time instead
    #    of the expected float32.  BASIC only does safe dead-code elimination and
    #    does NOT alter Cast nodes, so the model keeps its float32 I/O signature.
    try:
        opt_options = ort.SessionOptions()
        opt_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opt_options.optimized_model_filepath = onnx_output_path
        _ = ort.InferenceSession(onnx_output_path, opt_options, providers=['CPUExecutionProvider'])
        print(f"Successfully processed runtime optimizations on {onnx_output_path}")
    except Exception as e:
        print(f"Skipping generic ORT session optimization pass due to error: {e}")


def convert_realcugan_up2x_to_onnx(
    pth_path: Path,
    onnx_output_path: str = None,
    variant: str = 'full',
):
    print(f"Loading weights from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    state_dict = _extract_state_dict(checkpoint)
    if isinstance(state_dict, dict) and 'pro' in state_dict:
        state_dict = {k: v for k, v in state_dict.items() if k != 'pro'}

    if variant == 'full':
        model = UpCunet2x(in_channels=3, out_channels=3)
        strict = True
    elif variant == 'unet1_only':
        model = UpCunet2xUNet1Only(in_channels=3, out_channels=3)
        # Full checkpoints include unet2.* keys; ignore them for fast export.
        strict = False
    else:
        raise ValueError(f"Unsupported Real-CUGAN export variant: {variant}")

    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    if onnx_output_path is None:
        suffix = '' if variant == 'full' else '-unet1-only'
        onnx_output_path = f"{pth_path.stem}{suffix}.onnx"

    print(f"Exporting Real-CUGAN variant '{variant}' -> {onnx_output_path}")
    _export_dynamic_onnx(_ClampedModel(model), onnx_output_path)
    _postprocess_onnx_for_web(onnx_output_path)
    print(f"Conversion complete! Generated dynamic model: {onnx_output_path}")


def convert_realesrgan_x2plus_to_onnx():
    pth_path = Path("realesrgan_x2plus.pth")
    if not pth_path.exists():
        print(f"Skipping missing checkpoint: {pth_path}")
        return

    print(f"Loading weights from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    state_dict = _extract_state_dict(checkpoint)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    onnx_output_path = f"{pth_path.stem}.onnx"
    _export_dynamic_onnx(_ClampedModel(model), onnx_output_path)
    _postprocess_onnx_for_web(onnx_output_path)
    print(f"Conversion complete! Generated dynamic model: {onnx_output_path}")


def convert_realesr_animevideov3_to_onnx(pth_path: Path):
    if not pth_path.exists():
        print(f"Skipping missing checkpoint: {pth_path}")
        return

    print(f"Loading weights from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    state_dict = _extract_state_dict(checkpoint)

    # realesr-animevideov3 is a native x4 SRVGG model (16 conv blocks).
    # Export a 2x graph by adding a final 0.5x resize after native output.
    model = SRVGGNetCompact(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type='prelu',
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    model_2x = _ScaledOutputModel(model, scale_factor=0.5)
    onnx_output_path = f"{pth_path.stem}.onnx"
    _export_dynamic_onnx(_ClampedModel(model_2x), onnx_output_path)
    _postprocess_onnx_for_web(onnx_output_path)
    print(f"Conversion complete! Generated dynamic model: {onnx_output_path}")


def convert_mangajanai_to_onnx(pth_path: Path):
    if not pth_path.exists():
        print(f"Skipping missing checkpoint: {pth_path}")
        return

    print(f"Loading weights from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _remap_old_esrgan_keys(state_dict)

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    onnx_output_path = f"{pth_path.stem}.onnx"
    _export_dynamic_onnx(_ClampedModel(model), onnx_output_path)
    _postprocess_onnx_for_web(onnx_output_path)
    print(f"Conversion complete! Generated dynamic model: {onnx_output_path}")


def convert_mosr_gps_to_onnx(pth_path: Path):
    if not pth_path.exists():
        print(f"Skipping missing checkpoint: {pth_path}")
        return

    print(f"Loading weights from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    state_dict = _extract_state_dict(checkpoint)

    model = MoSRGPS(
        in_ch=3,
        out_ch=3,
        dim=64,
        n_block=24,
        upscale=2,
        kernel_size=7,
        expansion_ratio=1.5,
        conv_ratio=1.0,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    onnx_output_path = f"{pth_path.stem}.onnx"
    _export_dynamic_onnx(_ClampedModel(model), onnx_output_path)
    _postprocess_onnx_for_web(onnx_output_path)
    print(f"Conversion complete! Generated dynamic model: {onnx_output_path}")


def convert_realplksr_to_onnx(pth_path: Path):
    if not pth_path.exists():
        print(f"Skipping missing checkpoint: {pth_path}")
        return

    print(f"Loading weights from {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
    state_dict = _extract_state_dict(checkpoint)

    model = RealPLKSR(
        in_ch=3,
        out_ch=3,
        dim=64,
        n_blocks=28,
        upscaling_factor=2,
        kernel_size=17,
        split_ratio=0.25,
        use_ea=True,
        norm_groups=4,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    onnx_output_path = f"{pth_path.stem}.onnx"
    _export_dynamic_onnx(_ClampedModel(model), onnx_output_path)
    _postprocess_onnx_for_web(onnx_output_path)
    print(f"Conversion complete! Generated dynamic model: {onnx_output_path}")


if __name__ == "__main__":
    convert_realesrgan_x2plus_to_onnx()
    convert_realesr_animevideov3_to_onnx(Path("realesr-animevideov3.pth"))
    convert_mangajanai_to_onnx(Path("2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth"))
    convert_mosr_gps_to_onnx(Path("2x-AnimeSharpV2_MoSR_Sharp.pth"))
    convert_realplksr_to_onnx(Path("2x-AnimeSharpV2_RPLKSR_Sharp.pth"))
    for cugan_name in [
        "up2x-latest-conservative.pth",
        "up2x-latest-denoise1x.pth",
    ]:
        cugan_path = Path(cugan_name)
        if cugan_path.exists():
            # Export both quality path (UNet1+UNet2)
            # and fast path (UNet1 only).
            convert_realcugan_up2x_to_onnx(
                cugan_path,
                onnx_output_path=f"{cugan_path.stem}.onnx",
                variant='full',
            )
            convert_realcugan_up2x_to_onnx(
                cugan_path,
                onnx_output_path=f"{cugan_path.stem}-unet1-only.onnx",
                variant='unet1_only',
            )
        else:
            print(f"Skipping missing checkpoint: {cugan_path}")
