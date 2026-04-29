import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
)

# Channel dims for each ConvNeXt variant: (stage1, stage2, stage3, stage4)
CONVNEXT_CHANNELS = {
    "tiny":  (96,  192,  384,  768),
    "base":  (128, 256,  512,  1024),
    "large": (192, 384,  768,  1536),
}


def _load_convnext(variant: str, pretrained: bool):
    if variant == "tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        return convnext_tiny(weights=weights)
    elif variant == "base":
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        return convnext_base(weights=weights)
    elif variant == "large":
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        return convnext_large(weights=weights)
    else:
        raise ValueError(f"Unknown ConvNeXt variant: {variant}")


class ConvNeXtUNet(nn.Module):
    """U-Net with a ConvNeXt encoder for semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        variant: str = "tiny",
        pretrained: bool = True,
    ):
        super().__init__()
        c1, c2, c3, c4 = CONVNEXT_CHANNELS[variant]
        convnext = _load_convnext(variant, pretrained)

        # --- Encoder ---
        # features[0]: stem  (3 → c1, stride 4)
        # features[1]: stage-1 blocks (c1)
        # features[2]: downsample (c1 → c2, stride 2)
        # features[3]: stage-2 blocks (c2)
        # features[4]: downsample (c2 → c3, stride 2)
        # features[5]: stage-3 blocks (c3)
        # features[6]: downsample (c3 → c4, stride 2)
        # features[7]: stage-4 blocks (c4)
        self.encoder_stem   = convnext.features[0]  # c1, H/4
        self.encoder_stage1 = convnext.features[1]  # c1, H/4
        self.down2          = convnext.features[2]   # c2, H/8
        self.encoder_stage2 = convnext.features[3]  # c2, H/8
        self.down3          = convnext.features[4]   # c3, H/16
        self.encoder_stage3 = convnext.features[5]  # c3, H/16
        self.down4          = convnext.features[6]   # c4, H/32
        self.encoder_stage4 = convnext.features[7]  # c4, H/32

        # --- Decoder ---
        # up1: c4 → c3, then cat with x3 → 2*c3
        self.up1      = self._up_block(c4,     c3)
        self.dec_conv1 = self._conv_block(c3 * 2, c3)

        # up2: c3 → c2, then cat with x2 → 2*c2
        self.up2      = self._up_block(c3,     c2)
        self.dec_conv2 = self._conv_block(c2 * 2, c2)

        # up3: c2 → c1, then cat with x1 → 2*c1
        self.up3      = self._up_block(c2,     c1)
        self.dec_conv3 = self._conv_block(c1 * 2, c1)

        # up4: c1 → c1, then cat with x0_up → 2*c1
        self.up4      = self._up_block(c1,     c1)
        self.dec_conv4 = self._conv_block(c1 * 2, c1)

        # up5: c1 → 64, recover to full resolution
        self.up5      = self._up_block(c1, 64)

        # --- Segmentation head (raw logits, use cross-entropy loss) ---
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        # --- Encode ---
        x0 = self.encoder_stem(x)          # (B, c1, H/4,  W/4)
        x1 = self.encoder_stage1(x0)       # (B, c1, H/4,  W/4)
        x2 = self.encoder_stage2(self.down2(x1))  # (B, c2, H/8,  W/8)
        x3 = self.encoder_stage3(self.down3(x2))  # (B, c3, H/16, W/16)
        x4 = self.encoder_stage4(self.down4(x3))  # (B, c4, H/32, W/32)

        # --- Decode ---
        d = self.up1(x4)                            # (B, c3, H/16, W/16)
        d = self.dec_conv1(torch.cat([d, x3], 1))   # (B, c3, H/16, W/16)

        d = self.up2(d)                              # (B, c2, H/8, W/8)
        d = self.dec_conv2(torch.cat([d, x2], 1))   # (B, c2, H/8, W/8)

        d = self.up3(d)                              # (B, c1, H/4, W/4)
        d = self.dec_conv3(torch.cat([d, x1], 1))   # (B, c1, H/4, W/4)

        d = self.up4(d)                              # (B, c1, H/2, W/2)
        x0_up = F.interpolate(x0, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec_conv4(torch.cat([d, x0_up], 1))  # (B, c1, H/2, W/2)

        d = self.up5(d)                              # (B, 64, H, W)

        return self.seg_head(d)                      # (B, num_classes, H, W)


def build_model(
    in_channels: int = 3,
    num_classes: int = 2,
    variant: str = "tiny",
    pretrained: bool = True,
) -> ConvNeXtUNet:
    """Factory function — call with values from config.py."""
    return ConvNeXtUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
    )
