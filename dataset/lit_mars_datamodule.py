from io import BytesIO

import lightning as L
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from torch.utils.data import DataLoader

from dataset.ai4mars_dataset import AI4MarsDataset
from dataset.augmentation import build_transforms


def batch_validate_images(batch: list[dict]) -> list[bool]:
    """Fast batched validation of raw images."""
    results = []
    for image_dict in batch:
        try:
            raw_bytes = image_dict["bytes"]
            img = PILImage.open(BytesIO(raw_bytes))
            img.verify()
            results.append(True)
        except (UnidentifiedImageError, OSError, ValueError):
            results.append(False)
    return results


class LitMarsDataModule(L.LightningDataModule):
    def __init__(self, batch_size=8, num_workers=4, image_size=224, validation_split=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.validation_split = validation_split

    def prepare_data(self):
        self.dataset = load_dataset("gustavv-andrzejewski/ai4mars-terrain-segmentation", split="train")
        self.dataset = self.dataset.cast_column("image", HFImage(decode=False))
        self.dataset = self.dataset.filter(lambda x: x, input_columns=["has_labels"], num_proc=self.num_workers)
        self.dataset = self.dataset.filter(
            batch_validate_images,
            input_columns=["image"],
            batched=True,
            batch_size=1000,
            num_proc=self.num_workers,
        )
        self.dataset = self.dataset.cast_column("image", HFImage(decode=True))

    def setup(self, stage=None):
        train_tfms, val_tfms = build_transforms((self.image_size, self.image_size))

        split = self.dataset.train_test_split(test_size=self.validation_split, seed=42)
        self.train_dataset = AI4MarsDataset(split["train"], transform=train_tfms)
        self.val_dataset = AI4MarsDataset(split["test"], transform=val_tfms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
