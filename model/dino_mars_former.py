from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Core utilities
def create_activation(name: Literal["gelu", "relu", "leaky_relu"]) -> nn.Module:
    """Creates the specified activation function."""
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ChannelNorm(nn.Module):
    """Layer normalization applied to channel dimension in a [B, C, H, W] tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rearrange to normalize over channels, then restore original shape
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] → [B, H, W, C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style residual block with LayerScale."""

    def __init__(
        self,
        dim: int,
        activation: str = "gelu",
        expansion_ratio: int = 4,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()

        hidden_dim = dim * expansion_ratio

        # Block components
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.normalization = ChannelNorm(dim)
        self.pointwise_expand = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.activation = create_activation(activation)
        self.pointwise_reduce = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        # Learnable scaling factor (starts near zero)
        self.layer_scale = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Main branch
        y = self.depthwise_conv(x)
        y = self.normalization(y)
        y = self.pointwise_expand(y)
        y = self.activation(y)
        y = self.pointwise_reduce(y)

        # Apply scaling and add residual connection
        y = self.layer_scale.view(1, -1, 1, 1) * y
        return residual + y


class ConvNeXtDecoder(nn.Module):
    """Decoder using ConvNeXt blocks with residual connections and LayerScale."""

    def __init__(
        self,
        input_channels: Sequence[int],
        embed_dim: int,
        num_classes: int,
        output_size: Optional[tuple[int, int]],
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.output_size = output_size

        # Feature projection modules (one per input feature map)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, embed_dim, kernel_size=1, bias=False),
                    ConvNeXtBlock(embed_dim, activation=activation),
                )
                for channels in input_channels
            ]
        )

        # Feature fusion and prediction head
        self.fusion_head = nn.Sequential(
            nn.Conv2d(embed_dim * len(input_channels), embed_dim, kernel_size=1, bias=False),
            ConvNeXtBlock(embed_dim, activation=activation),
            nn.Dropout(0.1),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = self.output_size or features[0].shape[2:]

        # Project and upsample each feature level
        upsampled_features = [
            F.interpolate(proj(feat), size=target_size, mode="bilinear", align_corners=False)
            for proj, feat in zip(self.projections, features)
        ]

        # Concatenate and process through fusion head
        concatenated = torch.cat(upsampled_features, dim=1)
        return self.fusion_head(concatenated)


class MLPDecoder(nn.Module):
    """SegFormer-style decoder using only MLP/Conv1x1 operations."""

    def __init__(
        self,
        input_channels: Sequence[int],
        embed_dim: int,
        num_classes: int,
        output_size: Optional[tuple[int, int]],
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.output_size = output_size

        # Simple projection for each input feature
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, embed_dim, kernel_size=1, bias=True),
                    create_activation(activation),
                )
                for channels in input_channels
            ]
        )

        # Fusion layer and classifier
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(embed_dim * len(input_channels), embed_dim, kernel_size=1, bias=True),
            create_activation(activation),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1, bias=True)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = self.output_size or features[0].shape[2:]

        # Project and upsample each feature level
        upsampled_features = [
            F.interpolate(proj(feat), size=target_size, mode="bilinear", align_corners=False)
            for proj, feat in zip(self.projections, features)
        ]

        # Fuse features and classify
        fused = self.fusion_layer(torch.cat(upsampled_features, dim=1))
        return self.classifier(fused)


class DinoMarsFormer(nn.Module):
    """Mars terrain segmentation model combining DINOv2 backbone with a lightweight decoder."""

    def __init__(
        self,
        backbone: nn.Module,
        selected_layers: Sequence[int],
        embed_dim: int,
        num_classes: int,
        image_size: int,
        decoder_type: Literal["convnext", "mlp"] = "convnext",
        activation: Literal["gelu", "relu", "leaky_relu"] = "gelu",
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Backbone setup
        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.frozen_backbone = freeze_backbone

        # Feature extraction configuration
        self.selected_indices = [layer - 1 for layer in selected_layers]  # Convert to 0-indexed
        self.patch_size = 14  # ViT-S/14
        self.feature_size = image_size // self.patch_size
        self.image_size = image_size

        # Decoder setup
        backbone_dim = self.backbone.embed_dim
        input_channels = [backbone_dim] * len(selected_layers)
        output_size = (image_size // 4, image_size // 4)

        # Initialize appropriate decoder
        if decoder_type == "convnext":
            self.decoder: nn.Module = ConvNeXtDecoder(
                input_channels=input_channels,
                embed_dim=embed_dim,
                num_classes=num_classes,
                output_size=output_size,
                activation=activation,
            )
        elif decoder_type == "mlp":
            self.decoder = MLPDecoder(
                input_channels=input_channels,
                embed_dim=embed_dim,
                num_classes=num_classes,
                output_size=output_size,
                activation=activation,
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from selected transformer layers."""
        batch_size = x.shape[0]
        feature_h = feature_w = self.feature_size

        # Extract intermediate features with appropriate gradient context
        grad_context = torch.no_grad if self.frozen_backbone else torch.enable_grad
        with grad_context():
            intermediates = self.backbone.get_intermediate_layers(x, n=max(self.selected_indices) + 1)

        # Select and reshape features to spatial format
        selected_features = [intermediates[idx] for idx in self.selected_indices]
        feature_maps = [
            tokens.permute(0, 2, 1).reshape(batch_size, tokens.shape[-1], feature_h, feature_w)
            for tokens in selected_features
        ]

        return feature_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through backbone and decoder to generate segmentation output."""
        features = self.extract_features(x)
        return self.decoder(features)

    def count_parameters(self) -> None:
        """Print parameter statistics for the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")  # noqa
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")  # noqa
        print(f"Frozen parameters: {frozen_params:,} ({frozen_params/1e6:.1f}M)")  # noqa


if __name__ == "__main__":
    backbone_name = "dinov2_vits14"
    dino_backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)

    model = DinoMarsFormer(
        backbone=dino_backbone,
        selected_layers=[3, 6, 9, 11],
        embed_dim=256,
        num_classes=4,
        image_size=224,
        decoder_type="convnext",
        freeze_backbone=True,
    )

    model.count_parameters()

    sample_input = torch.randn(1, 3, 224, 224)
    logits = model(sample_input)

    print("Logits shape:", logits.shape)

    labels = torch.randint(0, 4, (1, 224, 224))
    labels_downsampled = F.interpolate(labels.unsqueeze(1).float(), (56, 56), mode="nearest").long().squeeze(1)

    loss = F.cross_entropy(logits, labels_downsampled)
    print("Training Loss:", loss.item())

    preds = logits.argmax(dim=1, keepdim=True).float()
    preds_full_res = F.interpolate(preds, (224, 224), mode="nearest").long()

    print("Upsampled preds shape:", preds_full_res.shape)
