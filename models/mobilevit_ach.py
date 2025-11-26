# mobilevit_single_file.py
# A self-contained implementation of MobileViT (https://arxiv.org/abs/2110.02178)
# Only depends on PyTorch

import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Dict, Tuple, Optional, Union

# from ..modules.block import AdaptiveCrossHadamard, DySoft, AdaptiveBottleneck
from .ach_bnc import AdaptiveBottleneckBNC


# ---------------------------------------------------------
# Extra Block
# ---------------------------------------------------------
            
# class AdaptiveBottleneckBNC(nn.Module):
#     def __init__(self, in_channel: int, ex_channel: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.ci = in_channel
#         self.ab = AdaptiveBottleneck(in_channel, in_channel, 'Hada', ex_channel, 3, 1, "DySoft")
        
#     def forward(self, x: Tensor):
#         B, N, C = x.shape
#         x = x.view(B, C, N, 1)
#         x = self.ab(x)
#         x = x.view(B, N, C)
#         return x
    
#     @staticmethod
#     def get_ex_channel(in_channel, expect_channel: int) -> int:
#         expect_channel -= in_channel
#         discriminant = 1 + 8*expect_channel
        
#         x1 = (1 + math.sqrt(discriminant)) / 2
#         x2 = (1 - math.sqrt(discriminant)) / 2
        
#         return math.floor(max(x1, x2))

# ---------------------------------------------------------
# Basic building blocks (Conv, BN, Act, etc.)
# ---------------------------------------------------------

class ConvLayer2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_norm: bool = True,
        use_act: bool = True,
        norm_layer: nn.Module = None,
        act_layer: nn.Module = None,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 if isinstance(kernel_size, int) else (
                (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2
            )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.norm = norm_layer(out_channels) if use_norm and norm_layer else nn.Identity()
        self.act = act_layer() if use_act and act_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        dilation: int = 1,
    ):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvLayer2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    use_norm=True,
                    use_act=True,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.SiLU,  # SiLU = Swish
                )
            )
        # dw
        layers.append(
            ConvLayer2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
                dilation=dilation,
                use_norm=True,
                use_act=True,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.SiLU,
            )
        )
        # pw-linear
        layers.append(
            ConvLayer2d(
                hidden_dim,
                out_channels,
                kernel_size=1,
                use_norm=True,
                use_act=False,
                norm_layer=nn.BatchNorm2d,
            )
        )

        self.conv = nn.Sequential(*layers)
        self.use_res_connect = use_res_connect

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ---------------------------------------------------------
# Attention & Transformer blocks
# ---------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * num_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.out_proj(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_ach: bool = False,
    ):
        super().__init__()
        self.pre_norm_mha = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=attn_dropout,
            bias=True,
        )

        self.pre_norm_cfl = nn.LayerNorm(embed_dim)
        self.cfl = AdaptiveBottleneckBNC(
            embed_dim, 
            AdaptiveBottleneckBNC.get_ex_channel(embed_dim, ffn_dim)
        ) if use_ach else nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm + MHA
        x = x + self.mha(self.pre_norm_mha(x))
        # Pre-norm + Channel Fusion
        x = x + self.cfl(self.pre_norm_cfl(x))
        return x


# ---------------------------------------------------------
# MobileViT Block
# ---------------------------------------------------------

class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        patch_h: int = 2,
        patch_w: int = 2,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        head_dim: int = 32,
        no_fusion: bool = False,
        conv_ksize: int = 3,
        use_ach = False,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = patch_h * patch_w
        self.no_fusion = no_fusion

        # local representation
        self.local_rep = nn.Sequential(
            ConvLayer2d(
                in_channels,
                in_channels,
                kernel_size=conv_ksize,
                groups=in_channels,
                use_norm=True,
                use_act=True,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.SiLU,
            ),
            ConvLayer2d(
                in_channels,
                transformer_dim,
                kernel_size=1,
                use_norm=False,
                use_act=False,
            ),
        )

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout,
                use_ach=use_ach
            )
            for _ in range(n_transformer_blocks)
        ]
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = ConvLayer2d(
            transformer_dim,
            in_channels,
            kernel_size=1,
            use_norm=True,
            use_act=True,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.SiLU,
        )

        self.fusion = None
        if not no_fusion:
            self.fusion = ConvLayer2d(
                2 * in_channels,
                in_channels,
                kernel_size=conv_ksize,
                use_norm=True,
                use_act=True,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.SiLU,
            )

    def unfolding(self, x):
        patch_h, patch_w = self.patch_h, self.patch_w
        B, C, H, W = x.shape

        # Compute padded size
        new_h = math.ceil(H / patch_h) * patch_h
        new_w = math.ceil(W / patch_w) * patch_w

        # Pad only if needed
        if new_h != H or new_w != W:
            x = torch.nn.functional.pad(x, (0, new_w - W, 0, new_h - H))

        # Reshape to patches: (B, C, new_h, new_w) -> (B * num_patches, patch_area, C)
        x = x.reshape(B, C, new_h // patch_h, patch_h, new_w // patch_w, patch_w)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(B, C, patch_h * patch_w, -1)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B, N, P, C)
        x = x.reshape(B * (new_h // patch_h) * (new_w // patch_w), self.patch_area, C)

        # ✅ Return: patches + (original H, W) for correct cropping later
        return x, (B, C, H, W)  # ← critical: store ORIGINAL H, W


    def folding(self, x, info):
        B, C, H_orig, W_orig = info  # ✅ now original size
        patch_h, patch_w = self.patch_h, self.patch_w
        N = (math.ceil(H_orig / patch_h)) * (math.ceil(W_orig / patch_w))

        # Reverse reshape
        x = x.view(B, N, self.patch_area, C)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.reshape(B, C, patch_h, patch_w, math.ceil(H_orig / patch_h), math.ceil(W_orig / patch_w))
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.reshape(B, C, math.ceil(H_orig / patch_h) * patch_h, math.ceil(W_orig / patch_w) * patch_w)

        # Crop back to original spatial size
        x = x[:, :, :H_orig, :W_orig]
        return x

    def forward(self, x):
        res = x

        fm = self.local_rep(x)
        # transformer global modeling
        patches, info = self.unfolding(fm)
        patches = self.global_rep(patches)
        fm = self.folding(patches, info)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        else:
            fm = fm + res

        return fm


# ---------------------------------------------------------
# Global Pooling
# ---------------------------------------------------------

class GlobalPool(nn.Module):
    def __init__(self, pool_type: str = "mean", keep_dim: bool = False):
        super().__init__()
        self.pool_type = pool_type.lower()
        self.keep_dim = keep_dim

    def forward(self, x):
        if self.pool_type == "mean":
            x = torch.mean(x, dim=[-2, -1], keepdim=self.keep_dim)
        elif self.pool_type == "max":
            x = torch.max(x, dim=-1, keepdim=self.keep_dim)[0]
            x = torch.max(x, dim=-2, keepdim=self.keep_dim)[0]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")
        return x


# ---------------------------------------------------------
# Configuration (small / x_small / xx_small)
# ---------------------------------------------------------

def get_mobilevit_config(mode: str = "small") -> Dict:
    if mode == "xx_small":
        # MobileViT-XXS
        return {
            "layer1": {"out_channels": 16, "num_blocks": 1, "stride": 1, "block_type": "mobilenet"},
            "layer2": {"out_channels": 24, "num_blocks": 3, "stride": 2, "block_type": "mobilenet"},
            "layer3": {
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 2,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": False
            },
            "layer4": {
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 2,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": True
            },
            "layer5": {
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 2,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": False
            },
            "last_layer_exp_factor": 4,
        }
    elif mode == "x_small":
        # MobileViT-XS
        return {
            "layer1": {"out_channels": 32, "num_blocks": 1, "stride": 1, "block_type": "mobilenet"},
            "layer2": {"out_channels": 48, "num_blocks": 3, "stride": 2, "block_type": "mobilenet"},
            "layer3": {
                "out_channels": 64,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 4,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": False
            },
            "layer4": {
                "out_channels": 80,
                "transformer_channels": 120,
                "ffn_dim": 240,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 4,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": True
            },
            "layer5": {
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 4,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": False
            },
            "last_layer_exp_factor": 4,
        }
    else:  # default: "small"
        return {
            "layer1": {"out_channels": 32, "num_blocks": 1, "stride": 1, "block_type": "mobilenet"},
            "layer2": {"out_channels": 64, "num_blocks": 3, "stride": 2, "block_type": "mobilenet"},
            "layer3": {
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 4,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": False
            },
            "layer4": {
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 4,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": True
            },
            "layer5": {
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": 4,
                "block_type": "mobilevit",
                "num_heads": 4,
                "use_ach": False
            },
            "last_layer_exp_factor": 4,
        }


# ---------------------------------------------------------
# MobileViT with ACH Model
# ---------------------------------------------------------

class mobilevit_ach(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        mode: str = "small",
        classifier_dropout: float = 0.0,
        pool_type: str = "mean",
    ):
        super().__init__()
        self.mode = mode
        self.dilation = 1
        self.dilate_l4 = False
        self.dilate_l5 = False

        self.config = get_mobilevit_config(mode)
        image_channels = 3
        out_channels = 16 if mode == "xx_small" else 32

        # First conv layer
        self.conv_1 = ConvLayer2d(
            image_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.SiLU,
        )
        in_channels = out_channels  # after conv_1

        # Layer 1
        self.layer_1, out_channels = self._make_layer(input_channel=in_channels, cfg=self.config["layer1"])
        in_channels = out_channels  # ✅ update

        # Layer 2
        self.layer_2, out_channels = self._make_layer(input_channel=in_channels, cfg=self.config["layer2"])
        in_channels = out_channels  # ✅ update

        # Layer 3
        self.layer_3, out_channels = self._make_layer(input_channel=in_channels, cfg=self.config["layer3"])
        in_channels = out_channels  # ✅ update

        # Layer 4
        self.layer_4, out_channels = self._make_layer(input_channel=in_channels, cfg=self.config["layer4"], dilate=self.dilate_l4)
        in_channels = out_channels  # ✅ update

        # Layer 5
        self.layer_5, out_channels = self._make_layer(input_channel=in_channels, cfg=self.config["layer5"], dilate=self.dilate_l5)
        in_channels = out_channels  # ✅ CRITICAL: now in_channels = layer_5's output channels

        # Expansion layer
        exp_channels = min(self.config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer2d(
            in_channels=in_channels,   # ✅ correct: e.g., 160 for MobileViT-small
            out_channels=exp_channels,
            kernel_size=1,
            use_norm=True,
            use_act=True,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.SiLU,
        )

        # Classifier
        self.classifier = nn.Sequential(
            GlobalPool(pool_type=pool_type, keep_dim=False),
            nn.Dropout(classifier_dropout) if 0.0 < classifier_dropout < 1.0 else nn.Identity(),
            nn.Linear(exp_channels, num_classes),
        )

    def _make_layer(
        self,
        input_channel: int,
        cfg: Dict,
        dilate: bool = False,
    ) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type == "mobilenet":
            return self._make_mobilenet_layer(input_channel, cfg)
        else:
            return self._make_mit_layer(input_channel, cfg, dilate)

    def _make_mobilenet_layer(
        self,
        input_channel: int,
        cfg: Dict,
    ) -> Tuple[nn.Sequential, int]:
        output_channels = cfg["out_channels"]
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)

        layers = []
        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            layers.append(layer)
            input_channel = output_channels
        return nn.Sequential(*layers), input_channel

    def _make_mit_layer(
        self,
        input_channel: int,
        cfg: Dict,
        dilate: bool = False,
    ) -> Tuple[nn.Sequential, int]:
        layers = []
        stride = cfg.get("stride", 1)

        # --- Step 1: Optional downsampling via IR block ---
        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1
            prev_dilation = self.dilation
            ir_layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg["out_channels"],
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
            )
            layers.append(ir_layer)
            input_channel = cfg["out_channels"]

        # --- Step 2: Prepare transformer parameters ---
        transformer_dim = cfg["transformer_channels"]  # ✅ explicitly defined
        ffn_dim = cfg["ffn_dim"]
        n_transformer_blocks = cfg.get("transformer_blocks", 2)
        patch_h = cfg.get("patch_h", 2)
        patch_w = cfg.get("patch_w", 2)

        # --- Step 3: Compute head_dim safely (CRITICAL FIX) ---
        # Prefer num_heads if provided; else default to 4
        num_heads = cfg.get("num_heads", 4)
        # Ensure head_dim is integer and divides transformer_dim
        if transformer_dim % num_heads != 0:
            # Adjust num_heads to largest divisor ≤ original
            for h in range(num_heads, 0, -1):
                if transformer_dim % h == 0:
                    num_heads = h
                    break
            print(f"[INFO] Adjusted num_heads to {num_heads} for transformer_dim={transformer_dim}")
        head_dim = transformer_dim // num_heads  # ✅ Now guaranteed integer

        # Final assertion (should never fail now)
        assert transformer_dim % head_dim == 0, \
            f"Failed to resolve head_dim: transformer_dim={transformer_dim}, head_dim={head_dim}"

        # --- Step 4: Add MobileViT block ---
        mobilevit_block = MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=n_transformer_blocks,
            patch_h=patch_h,
            patch_w=patch_w,
            dropout=0.1,
            ffn_dropout=0.0,
            attn_dropout=0.1,
            head_dim=head_dim,
            no_fusion=False,
            conv_ksize=3,
            use_ach=cfg.get("use_ach", False)
        )
        layers.append(mobilevit_block)

        return nn.Sequential(*layers), input_channel

    def forward_features(self, x):
        x = self.conv_1(x)  # stem
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------

if __name__ == "__main__":
    # Test all modes
    for mode in ["xx_small", "x_small", "small"]:
        print(f"\n=== Testing MobileViT-{mode.upper()} ===")
        model = mobilevit_ach(num_classes=1000, mode=mode)
        dummy_input = torch.randn(1, 3, 256, 256)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Example: export to ONNX (uncomment if needed)
    # torch.onnx.export(model, dummy_input, f"mobilevit_{mode}.onnx", opset_version=13)