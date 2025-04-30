from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightMLPDecoder(nn.Module):
    """Lightweight MLP Decoder inspired by SegFormer.

    Projects multi-scale features to a common dimension, upsamples to target resolution,
    concatenates, and applies final prediction layers.
    """

    def __init__(
        self,
        in_channels_list: list[int],
        embed_dim: int,
        num_classes: int,
        target_scale: Optional[tuple[int, int]] = None,
    ):
        super().__init__()

        self.proj_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                )
                for in_ch in in_channels_list
            ]
        )

        self.target_scale = target_scale

        self.fuse_layer = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = self.target_scale if self.target_scale else features[0].shape[2:]

        upsampled_features = []
        for proj_layer, feature_map in zip(self.proj_layers, features):
            projected_features = proj_layer(feature_map)
            upsampled = F.interpolate(projected_features, size=target_size, mode="bilinear", align_corners=False)
            upsampled_features.append(upsampled)

        fused_features = torch.cat(upsampled_features, dim=1)
        segmentation_logits = self.fuse_layer(fused_features)

        return segmentation_logits


class DinoMarsFormer(nn.Module):
    """Segmentation model combining DINOv2 transformer backbone with lightweight decoder.

    Extracts features from selected transformer layers and processes them for
    Mars terrain segmentation.
    """

    def __init__(
        self,
        dino_backbone: nn.Module,
        selected_layers: list[int],
        embed_dim: int,
        num_classes: int,
        image_size: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = dino_backbone

        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.frozen_backbone = freeze_backbone

        self.selected_layers_indices = [layer_num - 1 for layer_num in selected_layers]
        self.patch_size = 14
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        backbone_embed_dim = self.backbone.embed_dim
        in_channels = [backbone_embed_dim] * len(selected_layers)

        self.decoder = LightweightMLPDecoder(
            in_channels_list=in_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            target_scale=(image_size // 4, image_size // 4),
        )

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract feature maps from selected transformer layers."""
        batch_size = x.shape[0]
        feat_height = self.image_size // self.patch_size
        feat_width = feat_height

        if self.frozen_backbone:
            with torch.no_grad():
                intermediate_features = self.backbone.get_intermediate_layers(
                    x, n=max(self.selected_layers_indices) + 1
                )
        else:
            intermediate_features = self.backbone.get_intermediate_layers(
                x, n=max(self.selected_layers_indices) + 1
            )

        selected_features = [intermediate_features[i] for i in self.selected_layers_indices]

        feature_maps = [
            tokens.permute(0, 2, 1).reshape(batch_size, tokens.shape[-1], feat_height, feat_width)
            for tokens in selected_features
        ]

        return feature_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through backbone and decoder to generate segmentation output."""
        feature_maps = self.extract_features(x)
        segmentation_output = self.decoder(feature_maps)
        return segmentation_output

    def count_parameters(self) -> dict[str, int]:
        """Count and display trainable vs frozen parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Backbone parameters: {backbone_params:,} ({'frozen' if self.frozen_backbone else 'trainable'})")
        print(f"Decoder parameters: {decoder_params:,} (trainable)")

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "backbone": backbone_params,
            "decoder": decoder_params,
        }


if __name__ == "__main__":
    backbone_name = "dinov2_vits14"
    dino_backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)

    model = DinoMarsFormer(
        dino_backbone=dino_backbone,
        selected_layers=[3, 6, 9, 11],
        embed_dim=256,
        num_classes=4,
        image_size=224,
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
