from typing import Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import TERRAIN_CLASSES, IGNORE_INDEX, LitMarsDataModule
from model import DinoMarsFormer, LitDinoMarsFormer


def main(
    batch_size: int = 128,
    image_size: int = 224,
    validation_split: float = 0.1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    num_epochs: int = 200,
    decoder_type: Literal["convnext", "mlp"] = "convnext",
    decoder_embedding_dim: int = 56,
    decoder_activation: Literal["gelu", "relu", "leaky_relu"] = "gelu",
    selected_layers: list = [3, 6, 9, 12],
    backbone_name: str = "dinov2_vits14",
    save_dir: str = "checkpoints/",
    patience: int = 30,
):
    print(f"Loading backbone: {backbone_name}...")
    dino_backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)

    model = DinoMarsFormer(
        backbone=dino_backbone,
        selected_layers=selected_layers,
        embed_dim=decoder_embedding_dim,
        num_classes=len(TERRAIN_CLASSES),
        image_size=image_size,
        decoder_type=decoder_type,
        activation=decoder_activation,
        freeze_backbone=True,
    )

    lit_model = LitDinoMarsFormer(
        model=model,
        num_classes=len(TERRAIN_CLASSES),
        ignore_index=IGNORE_INDEX,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    data_module = LitMarsDataModule(
        batch_size=batch_size,
        num_workers=4,
        image_size=image_size,
        validation_split=validation_split,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        save_top_k=1,
        dirpath=save_dir,
        filename="best-checkpoint",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_iou",
        mode="max",
        patience=patience,
        verbose=True,
    )

    logger = TensorBoardLogger("lightning_logs", name="mars_segmentation")

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=num_epochs,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=20,
    )

    trainer.fit(lit_model, datamodule=data_module)


if __name__ == "__main__":
    main()
