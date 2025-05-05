import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.segmentation import DiceScore, GeneralizedDiceScore, MeanIoU
from typing import Dict, Tuple, Any

# Class labels for Mars terrain segmentation
TERRAIN_CLASSES = {
    0: "Soil",
    1: "Bedrock",
    2: "Sand",
    3: "Big Rock",
    4: "Null",
}


class LitDinoMarsFormer(L.LightningModule):
    """Lightning module for Mars terrain segmentation with DINOv2 backbone."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        ignore_index: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        # Initialize metrics
        self._init_metrics(num_classes, ignore_index)

    def _init_metrics(self, num_classes: int, ignore_index: int) -> None:
        """Initialize training and validation metrics."""
        # Accuracy metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index)

        # Segmentation metrics for validation
        common_args = {"num_classes": num_classes, "include_background": True, "input_format": "index"}

        self.val_iou = MeanIoU(
            **common_args,
            per_class=True,
        )

        self.val_dice_macro = DiceScore(
            **common_args,
            average="macro",
        )

        self.val_dice_weighted = DiceScore(
            **common_args,
            average="weighted",
        )

        self.val_generalized_dice = GeneralizedDiceScore(
            **common_args,
            per_class=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def shared_step(
        self, batch: Dict[str, torch.Tensor], stage: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared step for both training and validation."""
        images, masks = batch["image"], batch["mask"]
        logits = self(images)

        # Resize masks to match logits resolution
        downscaled_masks = (
            F.interpolate(masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
        )

        # Compute predictions and loss
        predictions = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, downscaled_masks)

        # Track accuracy
        accuracy = (
            self.train_acc(predictions, downscaled_masks)
            if stage == "train"
            else self.val_acc(predictions, downscaled_masks)
        )

        # Log metrics
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", accuracy, on_step=(stage == "train"), on_epoch=True, prog_bar=True)

        # Log learning rate during training
        if stage == "train":
            self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=False)

        return loss, predictions, downscaled_masks

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, _, _ = self.shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with additional metrics."""
        loss, predictions, targets = self.shared_step(batch, stage="val")
        self._compute_validation_metrics(predictions, targets)
        return loss

    def _compute_validation_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Compute and log detailed validation metrics."""
        # Filter out ignored regions
        ignore_index = self.hparams.ignore_index
        valid_mask = targets != ignore_index
        filtered_preds = predictions[valid_mask]
        filtered_targets = targets[valid_mask]

        # Add batch dimension for metrics
        filtered_preds = filtered_preds.unsqueeze(0)
        filtered_targets = filtered_targets.unsqueeze(0)

        # Compute segmentation metrics
        per_class_iou = self.val_iou(filtered_preds, filtered_targets)
        dice_macro = self.val_dice_macro(filtered_preds, filtered_targets)
        dice_weighted = self.val_dice_weighted(filtered_preds, filtered_targets)
        per_class_gds = self.val_generalized_dice(filtered_preds, filtered_targets)

        # Log overall metrics
        self.log("val_iou", per_class_iou.mean(), prog_bar=True, sync_dist=True)
        self.log("val_dice_macro", dice_macro, prog_bar=True, sync_dist=True)
        self.log("val_dice_weighted", dice_weighted, prog_bar=True, sync_dist=True)
        self.log("val_generalized_dice", per_class_gds.mean(), prog_bar=True, sync_dist=True)

        # Log per-class metrics (excluding ignored class)
        for class_idx, class_name in TERRAIN_CLASSES.items():
            if class_idx == ignore_index:
                continue
            self.log(f"val_iou_{class_name.lower()}", per_class_iou[class_idx], prog_bar=False, sync_dist=True)
            self.log(f"val_gds_{class_name.lower()}", per_class_gds[class_idx], prog_bar=False, sync_dist=True)
