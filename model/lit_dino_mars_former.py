import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.segmentation import DiceScore, GeneralizedDiceScore, MeanIoU

CLASS_LABELS = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock", 4: "Null"}


class LitDinoMarsFormer(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        ignore_index: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index)

        self.val_iou = MeanIoU(
            num_classes=num_classes,
            include_background=True,
            per_class=True,
            input_format="index",
        )
        self.val_dice = DiceScore(
            num_classes=num_classes,
            average="macro",
            include_background=True,
            input_format="index",
        )
        self.val_generalized_dice = GeneralizedDiceScore(
            num_classes=num_classes,
            include_background=True,
            per_class=True,
            input_format="index",
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def shared_step(self, batch, stage: str):
        imgs, masks = batch["image"], batch["mask"]
        logits = self(imgs)

        masks_down = (
            F.interpolate(masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
        )
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, masks_down)

        acc = self.train_acc(preds, masks_down) if stage == "train" else self.val_acc(preds, masks_down)
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=(stage == "train"), on_epoch=True, prog_bar=True)

        if stage == "train":
            self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=False)

        return loss, preds, masks_down

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.shared_step(batch, stage="val")
        self._log_validation_metrics(preds, targets)
        return loss

    def _log_validation_metrics(self, preds: torch.Tensor, targets: torch.Tensor):
        ignore_index = self.hparams.ignore_index

        # Valid regions only
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

        # Restore the original shape
        preds = preds.unsqueeze(0)
        targets = targets.unsqueeze(0)

        # Log average metrics
        mean_iou = self.val_iou(preds, targets)
        mean_dice = self.val_dice(preds, targets)
        classwise_gds = self.val_generalized_dice(preds, targets)

        self.log("val_iou", mean_iou.mean(), prog_bar=True, sync_dist=True)
        self.log("val_dice", mean_dice, prog_bar=True, sync_dist=True)
        self.log("val_generalized_dice", classwise_gds.mean(), prog_bar=True, sync_dist=True)

        # Per-class logging, excluding the 'Null' class (idx 4)
        for idx, class_name in CLASS_LABELS.items():
            if idx == ignore_index:
                continue
            self.log(f"val_iou_{class_name.lower()}", mean_iou[idx], prog_bar=False, sync_dist=True)
            self.log(f"val_gds_{class_name.lower()}", classwise_gds[idx], prog_bar=False, sync_dist=True)
