import numpy as np
import atexit



# =========================================================
# Debug log configuration
# =========================================================
LOG_FILE = "shape_debug.log"
_log_fp = open(LOG_FILE, "w", encoding="utf-8")

def log_print(*args):
    """
    Print to stdout and save the same message to the debug log file.
    """
    msg = " ".join(str(a) for a in args)
    print(msg)
    _log_fp.write(msg + "\n")
    _log_fp.flush()

def _close_log_file():
    try:
        _log_fp.close()
    except Exception:
        pass

atexit.register(_close_log_file)

# =========================================================
# Pure Python / NumPy inference-only implementation
# for QKFormer_10_384 / QKFormer_10_512 / QKFormer_10_768
# =========================================================
#
# Hardware-oriented notes:
#
# 1. This file is an explicit inference-only golden model.
#    It is written for readability and hardware mapping, not speed.
#
# 2. All parameter names keep the original PyTorch state_dict keys.
#    This is useful for:
#       - layer-by-layer validation
#       - direct mapping from training checkpoints
#       - easier debugging against the original model
#
# 3. BatchNorm uses running_mean / running_var only.
#    Therefore, BN here is inference BN, i.e. a fixed affine transform.
#
# 4. LIF uses soft reset:
#       v = v - spike * v_threshold
#    This behavior must be preserved if hardware needs bit-accurate
#    alignment with the reference model.
#
# 5. This model is unified and parameterized by embed_dim.
#    Supported model variants:
#
#       QKFormer_10_384:
#           embed_dim = 384
#           num_heads = 6
#           head_dim  = 64
#
#       QKFormer_10_512:
#           embed_dim = 512
#           num_heads = 8
#           head_dim  = 64
#
#       QKFormer_10_768:
#           embed_dim = 768
#           num_heads = 12
#           head_dim  = 64
#
# 6. Typical deployment assumptions for concrete shape comments:
#
#       input shape  = [B, 3, 224, 224]
#       typical B    = 1
#       typical T    = 4
#
#    Under this assumption, the spatial stages are typically:
#
#       input:        224 x 224
#       patch_embed1:  28 x 28
#       patch_embed2:  14 x 14
#       patch_embed3:   7 x 7
#
#    Token counts:
#
#       stage1 token count = 28 * 28 = 784
#       stage2 token count = 14 * 14 = 196
#       stage3 token count =  7 *  7 = 49
#
#    Note:
#       The exact channel counts in early patch embedding stages depend on
#       the trained checkpoint tensors. This code does not hardcode them.
#       Comments below focus on:
#           - concrete spatial sizes for 224x224 input
#           - concrete C / heads / head_dim in token-attention stages
#
# =========================================================


# ========================================================
# Basic tensor helpers
# ========================================================


def to_float32(x):
    """
    Convert input to float32.

    Typical example:
        input image:
            [1, 3, 224, 224] -> float32

    Hardware note:
        This is only a software-side datatype normalization.
        In hardware, this may later be replaced by:
            - FP16
            - BF16
            - INT8
            - fixed-point
    """
    return np.asarray(x, dtype=np.float32)


def flatten_01(x):
    """
    Merge dim-0 and dim-1.

    General:
        [T, B, ...] -> [T*B, ...]

    Typical example:
        [4, 1, 3, 224, 224] -> [4, 3, 224, 224]

    Why this is used:
        Many operators here are implemented in a standard batch form and do
        not explicitly support a separate time dimension. Therefore, [T, B]
        is flattened into a single batch-like dimension.

    Hardware interpretation:
        This corresponds to one of the following implementation strategies:
            - time as an outer loop
            - time-batch folding
            - time-multiplexed scheduling
    """
    s = x.shape
    return x.reshape((s[0] * s[1],) + s[2:])


def unflatten_01(x, t, b):
    """
    Split dim-0 back into [T, B].

    General:
        [T*B, ...] -> [T, B, ...]

    Typical example:
        [4, C, H, W] -> [4, 1, C, H, W] when t=4 and b=1

    Hardware interpretation:
        This is a logical shape restoration only.
        No arithmetic is involved.
    """
    s = x.shape
    return x.reshape((t, b) + s[1:])


# ========================================================
# Parameter loader
# ========================================================


class WeightStore:
    """
    Simple wrapper for NPZ weights.

    Expected behavior:
        - Load all parameters from NPZ.
        - Store them as float32.
        - Keep original PyTorch state_dict key names.

    Hardware analogy:
        This is the software equivalent of a parameter memory image.
    """

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.data = {k: data[k].astype(np.float32) for k in data.files}

    def get(self, key):
        """
        Fetch a tensor by key.

        Raises:
            KeyError if the key does not exist.
        """
        if key not in self.data:
            raise KeyError(f"Weight not found: {key}")
        return self.data[key]

    def has(self, key):
        """
        Check whether a given key exists.
        """
        return key in self.data


# ========================================================
# Primitive inference operators
# ========================================================


def batch_norm_2d_infer(x, weight, bias, running_mean, running_var, eps=1e-5):
    """
    Inference BatchNorm2d.

    Input:
        x           : [N, C, H, W]
        weight      : [C]
        bias        : [C]
        running_mean: [C]
        running_var : [C]

    Output:
        y           : [N, C, H, W]

    Formula:
        y = (x - mean) / sqrt(var + eps) * weight + bias

    Hardware note:
        In inference, BN is a fixed per-channel affine transform.
        It can often be folded into the previous Conv / Linear layer.
        Here it is kept explicit for correctness and verification.
    """
    weight = weight.reshape(1, -1, 1, 1)
    bias = bias.reshape(1, -1, 1, 1)
    running_mean = running_mean.reshape(1, -1, 1, 1)
    running_var = running_var.reshape(1, -1, 1, 1)
    return (x - running_mean) / np.sqrt(running_var + eps) * weight + bias


def batch_norm_1d_infer(x, weight, bias, running_mean, running_var, eps=1e-5):
    """
    Inference BatchNorm1d.

    Input:
        x           : [N, C, L]
        weight      : [C]
        bias        : [C]
        running_mean: [C]
        running_var : [C]

    Output:
        y           : [N, C, L]

    Typical usage in this file:
        - BN after 1D pointwise projections for Q / K / V / proj

    Hardware note:
        Same as BN2d: per-channel fixed affine transform.
    """
    weight = weight.reshape(1, -1, 1)
    bias = bias.reshape(1, -1, 1)
    running_mean = running_mean.reshape(1, -1, 1)
    running_var = running_var.reshape(1, -1, 1)
    return (x - running_mean) / np.sqrt(running_var + eps) * weight + bias


def conv1d_k1(x, weight, bias=None):
    """
    1D convolution with kernel_size=1, stride=1.

    Input:
        x      : [N, Cin, L]
        weight : [Cout, Cin, 1]
        bias   : [Cout] or None

    Output:
        y      : [N, Cout, L]

    Interpretation:
        Since kernel_size = 1, this is not a sliding convolution in L.
        It is equivalent to applying the same linear projection to every
        token / position independently.

    Hardware interpretation:
        This is a pointwise channel-mixing operator.
        It can be mapped as:
            - GEMM
            - matrix-vector multiply
            - 1x1 projection engine

    Typical attention-stage concrete examples:
        For stage2 (14x14):
            L = 196

        For stage3 (7x7):
            L = 49

        384 model:
            Cin = Cout = 384
            heads = 6
            head_dim = 64

        512 model:
            Cin = Cout = 512
            heads = 8
            head_dim = 64

        768 model:
            Cin = Cout = 768
            heads = 12
            head_dim = 64
    """
    w = weight[:, :, 0]
    y = np.einsum("ncl,oc->nol", x, w)
    if bias is not None:
        y = y + bias.reshape(1, -1, 1)
    return y.astype(np.float32)


def conv2d_general(x, weight, bias=None, stride=1, padding=0):
    """
    General Conv2d for inference.

    Input:
        x      : [N, Cin, H, W]
        weight : [Cout, Cin, Kh, Kw]
        bias   : [Cout] or None
        stride : int
        padding: int

    Output:
        y      : [N, Cout, Hout, Wout]

    Hout / Wout:
        Hout = (H + 2*padding - Kh) // stride + 1
        Wout = (W + 2*padding - Kw) // stride + 1

    Typical concrete examples:
        224x224 input with k=3, p=1, s=1:
            224x224 -> 224x224

        Then pooling with k=3, s=2, p=1:
            224x224 -> 112x112
            112x112 -> 56x56
            56x56   -> 28x28
            28x28   -> 14x14
            14x14   -> 7x7

    Hardware interpretation:
        This corresponds to a standard spatial convolution engine with:
            - input feature buffer
            - weight buffer
            - sliding window
            - MAC accumulation
    """
    n, cin, _, _ = x.shape
    cout, cin_w, kh, kw = weight.shape
    assert cin == cin_w, "Input channel mismatch"

    if padding > 0:
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )
    else:
        x_pad = x

    h_out = (x_pad.shape[2] - kh) // stride + 1
    w_out = (x_pad.shape[3] - kw) // stride + 1
    y = np.zeros((n, cout, h_out, w_out), dtype=np.float32)

    for oy in range(h_out):
        iy = oy * stride
        for ox in range(w_out):
            ix = ox * stride
            patch = x_pad[:, :, iy:iy + kh, ix:ix + kw]
            y[:, :, oy, ox] = np.einsum("nchw,ochw->no", patch, weight)

    if bias is not None:
        y += bias.reshape(1, -1, 1, 1)

    return y.astype(np.float32)


def maxpool2d_k3s2p1(x):
    """
    MaxPool2d with:
        kernel_size = 3
        stride      = 2
        padding     = 1

    Input:
        x: [N, C, H, W]

    Output:
        y: [N, C, Hout, Wout]

    Typical concrete sizes:
        224x224 -> 112x112
        112x112 -> 56x56
        56x56   -> 28x28
        28x28   -> 14x14
        14x14   -> 7x7

    Hardware interpretation:
        This is a fixed max-pooling operator, typically implemented using:
            - local window buffer
            - comparator tree
    """
    n, c, h, w = x.shape
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="constant",
        constant_values=-1e30,
    )
    h_out = (h + 2 - 3) // 2 + 1
    w_out = (w + 2 - 3) // 2 + 1

    y = np.zeros((n, c, h_out, w_out), dtype=np.float32)
    for oy in range(h_out):
        iy = oy * 2
        for ox in range(w_out):
            ix = ox * 2
            window = x_pad[:, :, iy:iy + 3, ix:ix + 3]
            y[:, :, oy, ox] = np.max(window, axis=(2, 3))
    return y


def linear(x, weight, bias):
    """
    Fully connected layer.

    Input:
        x      : [B, Cin] or [Cin]
        weight : [Cout, Cin]
        bias   : [Cout]

    Output:
        y      : [B, Cout] or [Cout]

    Typical final classifier example:
        input feature:
            [1, 384] / [1, 512] / [1, 768]
        output logits:
            [1, 1000]

    Hardware interpretation:
        Standard matrix-vector or matrix-matrix multiply.
    """
    return x @ weight.T + bias


def multistep_lif(x, tau=2.0, v_threshold=1.0):
    """
    Multi-step LIF inference with soft reset.

    Input:
        x: [T, ...]
    Output:
        out: [T, ...]

    Update rule:
        alpha = 1 / tau
        v = v + (x[t] - v) * alpha
        spike = (v >= v_threshold)
        v = v - spike * v_threshold

    Typical concrete example:
        If x is [4, 1, 384, 7, 7], then:
            - time steps = 4
            - state tensor v = [1, 384, 7, 7]

        If x is [4, 1, 6, 196, 64], then:
            - time steps = 4
            - state tensor v = [1, 6, 196, 64]

    Hardware interpretation:
        This is the key stateful neuron block.
        Each neuron needs:
            - membrane state storage
            - add/subtract datapath
            - threshold comparator
            - reset logic

    Important:
        This model uses soft reset, not hard reset.
        That difference matters for exact alignment.
    """
    x = x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    v = np.zeros_like(x[0], dtype=np.float32)

    alpha = 1.0 / tau

    for t in range(x.shape[0]):
        v = v + (x[t] - v) * alpha
        spike = (v >= v_threshold).astype(np.float32)
        out[t] = spike
        v = v - spike * v_threshold

    return out


# ========================================================
# Unified explicit model
# ========================================================


class QKFormer_Explicit:
    """
    Unified NumPy inference model for QKFormer depth-10 variants.

    Supported variants:
        - QKFormer_10_384
        - QKFormer_10_512
        - QKFormer_10_768

    Model structure:
        input
          -> patch_embed1
          -> stage1.0 token block
          -> patch_embed2
          -> stage2.0 token block
          -> stage2.1 token block
          -> patch_embed3
          -> stage3.x spiking blocks
          -> spatial average
          -> temporal average
          -> classification head

    Concrete model-specific channel specs:
        384 model:
            embed_dim = 384
            num_heads = 6
            head_dim  = 64

        512 model:
            embed_dim = 512
            num_heads = 8
            head_dim  = 64

        768 model:
            embed_dim = 768
            num_heads = 12
            head_dim  = 64
    """

    MODEL_CFG = {
        384: {"num_heads": 6, "depths": 10},
        512: {"num_heads": 8, "depths": 10},
        768: {"num_heads": 12, "depths": 10},
    }

    def __init__(self, npz_path, embed_dim=384, T=4, num_classes=1000):
        """
        Args:
            npz_path    : path to NPZ weight file
            embed_dim   : model width, one of {384, 512, 768}
            T           : number of time steps
            num_classes : output class count

        Concrete configurations:
            embed_dim=384:
                num_heads=6, head_dim=64

            embed_dim=512:
                num_heads=8, head_dim=64

            embed_dim=768:
                num_heads=12, head_dim=64

        Typical runtime example:
            input  = [1, 3, 224, 224]
            T      = 4
            output = [1, 1000]
        """
        if embed_dim not in self.MODEL_CFG:
            raise ValueError(f"Unsupported embed_dim: {embed_dim}")

        cfg = self.MODEL_CFG[embed_dim]
        self.ws = WeightStore(npz_path)
        self.embed_dim = embed_dim
        self.T = T
        self.num_classes = num_classes
        self.num_heads = cfg["num_heads"]
        self.depths = cfg["depths"]
        self.head_dim = embed_dim // self.num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.mlp_ratio = 4

    def __call__(self, x):
        return self.forward(x)

    def _conv1d_bias(self, key):
        """
        Return 1D conv bias if present, else None.
        """
        return self.ws.get(key) if self.ws.has(key) else None

    def _conv2d_bias(self, key):
        """
        Return 2D conv bias if present, else None.
        """
        return self.ws.get(key) if self.ws.has(key) else None

    def _bn2d(self, x, prefix):
        """
        Convenience wrapper for BN2d parameter lookup.
        """
        return batch_norm_2d_infer(
            x,
            self.ws.get(f"{prefix}.weight"),
            self.ws.get(f"{prefix}.bias"),
            self.ws.get(f"{prefix}.running_mean"),
            self.ws.get(f"{prefix}.running_var"),
        )

    def _bn1d(self, x, prefix):
        """
        Convenience wrapper for BN1d parameter lookup.
        """
        return batch_norm_1d_infer(
            x,
            self.ws.get(f"{prefix}.weight"),
            self.ws.get(f"{prefix}.bias"),
            self.ws.get(f"{prefix}.running_mean"),
            self.ws.get(f"{prefix}.running_var"),
        )


    def _shape_str(self, x):
        """
        Return shape string for debug printing.
        """
        return str(tuple(x.shape))

    def _debug_before_after(self, op_name, x_in, w_in=None, y_out=None, extra=None):
        """
        Print input/output shapes around selected operators.
        """
        log_print(f"[DEBUG] {op_name}")
        log_print(f"        input : {self._shape_str(x_in)}")
        if w_in is not None:
            log_print(f"        weight: {self._shape_str(w_in)}")
        if extra is not None:
            log_print(f"        extra : {extra}")
        if y_out is not None:
            log_print(f"        output: {self._shape_str(y_out)}")

    def _patch_embed1(self, x_t):
        """
        First patch embedding stage.

        Input:
            x_t: [T, B, 3, 224, 224]
            Typical example:
                [4, 1, 3, 224, 224]

        Main flow:
            1. proj_conv + BN + MaxPool + LIF
            2. proj1_conv + BN + MaxPool + LIF
            3. proj2_conv + BN + LIF
            4. residual path: res_conv(stride=2) + BN + LIF
            5. add main path and residual path

        Typical spatial evolution for 224x224 input:
            after first pool:  224 -> 112
            after second pool: 112 -> 56
            after residual stride2 on first pooled result:
                               112 -> 56

        Therefore, typical output spatial size:
            [T, B, C1, 56, 56]

        Notes:
            - Exact C1 depends on checkpoint weights.
            - This stage is shared structurally across 384/512/768 models.
            - Concrete embed_dim differences mainly become explicit in later
              token-attention stages.
        """
        t, b, _, h, w = x_t.shape

        y = conv2d_general(
            flatten_01(x_t),
            self.ws.get("patch_embed1.proj_conv.weight"),
            bias=self._conv2d_bias("patch_embed1.proj_conv.bias"),
            stride=1,
            padding=1,
        )
        y = self._bn2d(y, "patch_embed1.proj_bn")
        y = maxpool2d_k3s2p1(y)
        y = unflatten_01(y, t, b)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)

        x_res = flatten_01(y)

        y = conv2d_general(
            x_res,
            self.ws.get("patch_embed1.proj1_conv.weight"),
            bias=self._conv2d_bias("patch_embed1.proj1_conv.bias"),
            stride=1,
            padding=1,
        )
        y = self._bn2d(y, "patch_embed1.proj1_bn")
        y = maxpool2d_k3s2p1(y)
        y = unflatten_01(y, t, b)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)

        y = conv2d_general(
            flatten_01(y),
            self.ws.get("patch_embed1.proj2_conv.weight"),
            bias=self._conv2d_bias("patch_embed1.proj2_conv.bias"),
            stride=1,
            padding=1,
        )
        y = self._bn2d(y, "patch_embed1.proj2_bn")
        y = unflatten_01(y, t, b)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)

        y_res = conv2d_general(
            x_res,
            self.ws.get("patch_embed1.res_conv.weight"),
            bias=self._conv2d_bias("patch_embed1.res_conv.bias"),
            stride=2,
            padding=0,
        )
        y_res = self._bn2d(y_res, "patch_embed1.res_bn")
        y_res = unflatten_01(y_res, t, b)
        y_res = multistep_lif(y_res, tau=2.0, v_threshold=1.0)

        return y + y_res

    def _patch_embed_stage(self, x, prefix):
        """
        Generic later patch embedding stage.

        Input:
            x: [T, B, Cin, H, W]

        Flow:
            main path:
                proj3_conv + BN + MaxPool + LIF
                proj4_conv + BN + LIF

            residual path:
                res_conv(stride=2) + BN + LIF

            output:
                main + residual

        Typical spatial evolution for 224x224 input:
            stage after patch_embed1:
                56 -> 28

            next patch stage:
                28 -> 14

            next patch stage:
                14 -> 7

        So typical output spatial sizes are:
            patch_embed2 output: [T, B, C2, 14, 14]
            patch_embed3 output: [T, B, C3, 7, 7]

        Notes:
            - Exact channel counts are determined by the loaded weights.
            - Spatial sizes are concrete under 224x224 input.
        """
        t, b, _, _, _ = x.shape
        src = flatten_01(x)

        y = conv2d_general(
            src,
            self.ws.get(f"{prefix}.proj3_conv.weight"),
            bias=self._conv2d_bias(f"{prefix}.proj3_conv.bias"),
            stride=1,
            padding=1,
        )
        y = self._bn2d(y, f"{prefix}.proj3_bn")
        y = maxpool2d_k3s2p1(y)
        y = unflatten_01(y, t, b)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)

        y = conv2d_general(
            flatten_01(y),
            self.ws.get(f"{prefix}.proj4_conv.weight"),
            bias=self._conv2d_bias(f"{prefix}.proj4_conv.bias"),
            stride=1,
            padding=1,
        )
        y = self._bn2d(y, f"{prefix}.proj4_bn")
        y = unflatten_01(y, t, b)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)

        y_res = conv2d_general(
            src,
            self.ws.get(f"{prefix}.res_conv.weight"),
            bias=self._conv2d_bias(f"{prefix}.res_conv.bias"),
            stride=2,
            padding=0,
        )
        y_res = self._bn2d(y_res, f"{prefix}.res_bn")
        y_res = unflatten_01(y_res, t, b)
        y_res = multistep_lif(y_res, tau=2.0, v_threshold=1.0)

        return y + y_res

    def _mlp(self, x, prefix):
        """
        MLP block implemented as two Conv2d layers.

        Input:
            x: [T, B, C, H, W]

        Output:
            y: [T, B, C, H, W]

        Flow:
            fc1_conv + BN + LIF
            fc2_conv + BN + LIF

        Typical concrete examples:
            stage2 token block:
                x = [4, 1, 384/512/768, 14, 14]
                or
                x = [4, 1, 384/512/768, 7, 7]

        Important:
            Although the layer names contain "fc", the implementation is Conv2d.
            In many checkpoints this behaves like pointwise channel mixing.

        Hardware interpretation:
            Good candidate for pointwise MAC / GEMM-style implementation.
        """
        t, b, _, _, _ = x.shape

        y = conv2d_general(
            flatten_01(x),
            self.ws.get(f"{prefix}.fc1_conv.weight"),
            bias=self._conv2d_bias(f"{prefix}.fc1_conv.bias"),
            stride=1,
            padding=0,
        )
        y = self._bn2d(y, f"{prefix}.fc1_bn")
        y = unflatten_01(y, t, b)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)

        y = conv2d_general(
            flatten_01(y),
            self.ws.get(f"{prefix}.fc2_conv.weight"),
            bias=self._conv2d_bias(f"{prefix}.fc2_conv.bias"),
            stride=1,
            padding=0,
        )
        y = self._bn2d(y, f"{prefix}.fc2_bn")
        y = unflatten_01(y, t, b)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)
        return y

    def _token_qk_attn(self, x, prefix):
        """
        Lightweight token Q/K attention block.

        Input:
            x: [T, B, C, H, W]

        Typical concrete stage example:
            if H=W=14:
                token count N = 14 * 14 = 196

            if H=W=7:
                token count N = 7 * 7 = 49

        Concrete channel specs:
            384 model:
                C = 384
                heads = 6
                head_dim = 64

            512 model:
                C = 512
                heads = 8
                head_dim = 64

            768 model:
                C = 768
                heads = 12
                head_dim = 64

        Flow:
            1. Flatten spatial dims:
                   [T, B, C, H, W] -> [T, B, C, N]
            2. Q projection:
                   [T*B, C, N] -> [T, B, heads, head_dim, N]
            3. K projection:
                   [T*B, C, N] -> [T, B, heads, head_dim, N]
            4. Generate gate:
                   attn = sum(Q over head_dim)
            5. Spike gate using LIF with threshold 0.5
            6. Multiply gate with K
            7. Output projection back to [T, B, C, H, W]

        Important:
            This is not standard QK^T softmax attention.
            It is a simpler gating-style attention.

        Hardware interpretation:
            Cheaper than dense full attention because it avoids:
                - N x N score matrix
                - softmax
        """
        t, b, c, h, w = x.shape
        n = h * w
        head_dim = c // self.num_heads

        x_flat = x.reshape(t, b, c, n)
        x_qk = x_flat.reshape(t * b, c, n)

        q_weight = self.ws.get(f"{prefix}.q_conv.weight")
        self._debug_before_after(f"{prefix}.q_conv / conv1d_k1 (before)", x_qk, q_weight)
        q = conv1d_k1(
            x_qk,
            q_weight,
            bias=self._conv1d_bias(f"{prefix}.q_conv.bias"),
        )
        self._debug_before_after(f"{prefix}.q_conv / conv1d_k1 (after)", x_qk, q_weight, q)
        q = self._bn1d(q, f"{prefix}.q_bn")
        q = q.reshape(t, b, c, n)
        q = multistep_lif(q, tau=2.0, v_threshold=1.0)
        q = q.reshape(t, b, self.num_heads, head_dim, n)

        k_weight = self.ws.get(f"{prefix}.k_conv.weight")
        self._debug_before_after(f"{prefix}.k_conv / conv1d_k1 (before)", x_qk, k_weight)
        k = conv1d_k1(
            x_qk,
            k_weight,
            bias=self._conv1d_bias(f"{prefix}.k_conv.bias"),
        )
        self._debug_before_after(f"{prefix}.k_conv / conv1d_k1 (after)", x_qk, k_weight, k)
        k = self._bn1d(k, f"{prefix}.k_bn")
        k = k.reshape(t, b, c, n)
        k = multistep_lif(k, tau=2.0, v_threshold=1.0)
        k = k.reshape(t, b, self.num_heads, head_dim, n)

        attn = np.sum(q, axis=3, keepdims=True)
        attn = multistep_lif(attn, tau=2.0, v_threshold=0.5)

        y = attn * k
        y = y.reshape(t, b, c, n)

        y_proj_in = y.reshape(t * b, c, n)
        y_proj_weight = self.ws.get(f"{prefix}.proj_conv.weight")
        self._debug_before_after(f"{prefix}.proj_conv / conv1d_k1 (before)", y_proj_in, y_proj_weight)
        y = conv1d_k1(
            y_proj_in,
            y_proj_weight,
            bias=self._conv1d_bias(f"{prefix}.proj_conv.bias"),
        )
        self._debug_before_after(f"{prefix}.proj_conv / conv1d_k1 (after)", y_proj_in, y_proj_weight, y)
        y = self._bn1d(y, f"{prefix}.proj_bn")
        y = y.reshape(t, b, c, h, w)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)
        return y

    def _spiking_self_attn(self, x, prefix):
        """
        Spiking self-attention block.

        Input:
            x: [T, B, C, H, W]

        Typical concrete stage example:
            stage3 under 224x224 input often uses:
                H = 7
                W = 7
                token count N = 49

            If used at 14x14:
                N = 196

        Concrete model specs:
            384 model:
                C = 384
                heads = 6
                head_dim = 64

                reshaped Q/K/V:
                    [T, B, 6, N, 64]

            512 model:
                C = 512
                heads = 8
                head_dim = 64

                reshaped Q/K/V:
                    [T, B, 8, N, 64]

            768 model:
                C = 768
                heads = 12
                head_dim = 64

                reshaped Q/K/V:
                    [T, B, 12, N, 64]

        Core computation:
            kv = K^T @ V
            y  = Q @ kv

        Concrete matrix sizes:
            For 384 model:
                K^T @ V:
                    [T, B, 6, 64, N] @ [T, B, 6, N, 64]
                    -> [T, B, 6, 64, 64]

                Q @ kv:
                    [T, B, 6, N, 64] @ [T, B, 6, 64, 64]
                    -> [T, B, 6, N, 64]

            For 512 model:
                -> [T, B, 8, 64, 64]
                -> [T, B, 8, N, 64]

            For 768 model:
                -> [T, B, 12, 64, 64]
                -> [T, B, 12, N, 64]

        Important:
            This is not standard softmax(QK^T)V.
            It uses the reordered form:
                Q(K^T V)

        Hardware benefit:
            Avoids explicit N x N attention matrix.
            Since head_dim is always 64, the core per-head matrix size is fixed:
                64 x 64
            This is very hardware-friendly across all three model sizes.
        """
        t, b, c, h, w = x.shape
        n = h * w
        head_dim = c // self.num_heads
        scale = 1.0 / np.sqrt(head_dim)

        x_flat = x.reshape(t, b, c, n)
        x_qkv = x_flat.reshape(t * b, c, n)

        q_weight = self.ws.get(f"{prefix}.q_conv.weight")
        self._debug_before_after(f"{prefix}.q_conv / conv1d_k1 (before)", x_qkv, q_weight)
        q = conv1d_k1(
            x_qkv,
            q_weight,
            bias=self._conv1d_bias(f"{prefix}.q_conv.bias"),
        )
        self._debug_before_after(f"{prefix}.q_conv / conv1d_k1 (after)", x_qkv, q_weight, q)
        q = self._bn1d(q, f"{prefix}.q_bn")
        q = q.reshape(t, b, c, n)
        q = multistep_lif(q, tau=2.0, v_threshold=1.0)
        q = np.transpose(q, (0, 1, 3, 2)).reshape(t, b, n, self.num_heads, head_dim)
        q = np.transpose(q, (0, 1, 3, 2, 4))

        k_weight = self.ws.get(f"{prefix}.k_conv.weight")
        self._debug_before_after(f"{prefix}.k_conv / conv1d_k1 (before)", x_qkv, k_weight)
        k = conv1d_k1(
            x_qkv,
            k_weight,
            bias=self._conv1d_bias(f"{prefix}.k_conv.bias"),
        )
        self._debug_before_after(f"{prefix}.k_conv / conv1d_k1 (after)", x_qkv, k_weight, k)
        k = self._bn1d(k, f"{prefix}.k_bn")
        k = k.reshape(t, b, c, n)
        k = multistep_lif(k, tau=2.0, v_threshold=1.0)
        k = np.transpose(k, (0, 1, 3, 2)).reshape(t, b, n, self.num_heads, head_dim)
        k = np.transpose(k, (0, 1, 3, 2, 4))

        v_weight = self.ws.get(f"{prefix}.v_conv.weight")
        self._debug_before_after(f"{prefix}.v_conv / conv1d_k1 (before)", x_qkv, v_weight)
        v = conv1d_k1(
            x_qkv,
            v_weight,
            bias=self._conv1d_bias(f"{prefix}.v_conv.bias"),
        )
        self._debug_before_after(f"{prefix}.v_conv / conv1d_k1 (after)", x_qkv, v_weight, v)
        v = self._bn1d(v, f"{prefix}.v_bn")
        v = v.reshape(t, b, c, n)
        v = multistep_lif(v, tau=2.0, v_threshold=1.0)
        v = np.transpose(v, (0, 1, 3, 2)).reshape(t, b, n, self.num_heads, head_dim)
        v = np.transpose(v, (0, 1, 3, 2, 4))

        k_t = np.transpose(k, (0, 1, 2, 4, 3))
        log_print(f"[DEBUG] {prefix}.kv_matmul / before")
        log_print(f"        lhs   : {self._shape_str(k_t)}")
        log_print(f"        rhs   : {self._shape_str(v)}")
        kv = np.matmul(k_t, v)
        log_print(f"[DEBUG] {prefix}.kv_matmul / after")
        log_print(f"        output: {self._shape_str(kv)}")

        log_print(f"[DEBUG] {prefix}.qkv_matmul / before")
        log_print(f"        lhs   : {self._shape_str(q)}")
        log_print(f"        rhs   : {self._shape_str(kv)}")
        y = np.matmul(q, kv) * scale
        log_print(f"[DEBUG] {prefix}.qkv_matmul / after")
        log_print(f"        output: {self._shape_str(y)}")

        y = np.transpose(y, (0, 1, 3, 2, 4)).reshape(t, b, n, c)
        y = np.transpose(y, (0, 1, 3, 2)).reshape(t, b, c, n)
        y = multistep_lif(y, tau=2.0, v_threshold=0.5)

        y_proj_in = y.reshape(t * b, c, n)
        y_proj_weight = self.ws.get(f"{prefix}.proj_conv.weight")
        self._debug_before_after(f"{prefix}.proj_conv / conv1d_k1 (before)", y_proj_in, y_proj_weight)
        y = conv1d_k1(
            y_proj_in,
            y_proj_weight,
            bias=self._conv1d_bias(f"{prefix}.proj_conv.bias"),
        )
        self._debug_before_after(f"{prefix}.proj_conv / conv1d_k1 (after)", y_proj_in, y_proj_weight, y)
        y = self._bn1d(y, f"{prefix}.proj_bn")
        y = y.reshape(t, b, c, h, w)
        y = multistep_lif(y, tau=2.0, v_threshold=1.0)
        return y

    def _token_block(self, x, prefix):
        """
        Token block with residual structure.

        Structure:
            x = x + token_qk_attn(x)
            x = x + mlp(x)

        Typical concrete examples:
            stage1 or stage2:
                x may be [4, 1, 384/512/768, 14, 14]
                or    [4, 1, 384/512/768, 7, 7]

        Hardware interpretation:
            Standard residual block:
                - attention sub-block
                - residual add
                - MLP sub-block
                - residual add
        """
        x = x + self._token_qk_attn(x, f"{prefix}.attn")
        x = x + self._mlp(x, f"{prefix}.mlp")
        return x

    def _spiking_block(self, x, prefix):
        """
        Spiking attention block with residual structure.

        Structure:
            x = x + spiking_self_attn(x)
            x = x + mlp(x)

        Concrete model-channel cases:
            384 model:
                x = [T, B, 384, H, W]

            512 model:
                x = [T, B, 512, H, W]

            768 model:
                x = [T, B, 768, H, W]
        """
        x = x + self._spiking_self_attn(x, f"{prefix}.attn")
        x = x + self._mlp(x, f"{prefix}.mlp")
        return x

    def forward_features(self, x_t):
        """
        Backbone forward path.

        Input:
            x_t: [T, B, 3, 224, 224]
            Typical:
                [4, 1, 3, 224, 224]

        Stage flow:
            1. patch_embed1
            2. stage1.0 token block
            3. patch_embed2
            4. stage2.0 token block
            5. stage2.1 token block
            6. patch_embed3
            7. stage3.0 ... stage3.(depths-4) spiking blocks
            8. spatial average pooling

        Typical spatial evolution for 224x224 input:
            input           : 224 x 224
            after patch1    :  56 x 56
            after patch2    :  14 x 14
            after patch3    :   7 x  7

        Final output:
            [T, B, C]
            where C is:
                384 / 512 / 768
        """
        x = self._patch_embed1(x_t)
        x = self._token_block(x, "stage1.0")

        x = self._patch_embed_stage(x, "patch_embed2")
        x = self._token_block(x, "stage2.0")
        x = self._token_block(x, "stage2.1")

        x = self._patch_embed_stage(x, "patch_embed3")
        for i in range(self.depths - 3):
            x = self._spiking_block(x, f"stage3.{i}")

        t, b, c, h, w = x.shape

        # Spatial global average pooling:
        #   [T, B, C, H, W] -> [T, B, C, H*W] -> [T, B, C]
        #
        # Concrete example:
        #   if H=W=7, then H*W=49
        return x.reshape(t, b, c, h * w).mean(axis=3)

    def forward(self, x):
        """
        Top-level inference entry.

        Input:
            x: [B, 3, 224, 224]
            Typical:
                [1, 3, 224, 224]

        Flow:
            1. Convert to float32
            2. Repeat the static image across time:
                   [B, 3, 224, 224] -> [T, B, 3, 224, 224]
            3. Run backbone:
                   [T, B, C]
            4. Average over time:
                   [B, C]
            5. Final classifier:
                   [B, num_classes]

        Concrete outputs:
            384 model:
                feature before head = [1, 384]
                logits              = [1, 1000]

            512 model:
                feature before head = [1, 512]
                logits              = [1, 1000]

            768 model:
                feature before head = [1, 768]
                logits              = [1, 1000]

        Important:
            The input image is static across time steps.
            Temporal dynamics come from internal neuron states, not from
            changing input frames.
        """
        x_in = to_float32(x)
        x_t = np.repeat(x_in[np.newaxis, ...], self.T, axis=0)
        feat = self.forward_features(x_t)
        feat = feat.mean(axis=0)
        head_weight = self.ws.get("head.weight")
        log_print("[DEBUG] head.linear / before")
        log_print(f"        input : {self._shape_str(feat)}")
        log_print(f"        weight: {self._shape_str(head_weight)}")
        logits = linear(feat, head_weight, self.ws.get("head.bias"))
        log_print("[DEBUG] head.linear / after")
        log_print(f"        output: {self._shape_str(logits)}")
        return logits.astype(np.float32)


# ========================================================
# Convenience builders
# ========================================================


def QKFormer_10_384(npz_path, T=1, num_classes=1000):
    """
    Build QKFormer_10_384.

    Concrete configuration:
        embed_dim = 384
        num_heads = 6
        head_dim  = 64

    Typical classifier output:
        [B, 1000]
    """
    return QKFormer_Explicit(npz_path=npz_path, embed_dim=384, T=T, num_classes=num_classes)


def QKFormer_10_512(npz_path, T=1, num_classes=1000):
    """
    Build QKFormer_10_512.

    Concrete configuration:
        embed_dim = 512
        num_heads = 8
        head_dim  = 64

    Typical classifier output:
        [B, 1000]
    """
    return QKFormer_Explicit(npz_path=npz_path, embed_dim=512, T=T, num_classes=num_classes)


def QKFormer_10_768(npz_path, T=1, num_classes=1000):
    """
    Build QKFormer_10_768.

    Concrete configuration:
        embed_dim = 768
        num_heads = 12
        head_dim  = 64

    Typical classifier output:
        [B, 1000]
    """
    return QKFormer_Explicit(npz_path=npz_path, embed_dim=768, T=T, num_classes=num_classes)