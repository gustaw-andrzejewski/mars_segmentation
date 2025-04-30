from typing import Callable, Optional

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

CLASS_LABELS = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock", 255: "Null"}
IGNORE_INDEX = 4


class AI4MarsDataset(Dataset):
    def __init__(self, hf_dataset: HFDataset, transform: Optional[Callable] = None, ignore_null: bool = True):
        self.ds = hf_dataset
        self.tfm = transform
        self.ignore_null = ignore_null

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.ds[idx]
        img = np.asarray(ex["image"].convert("RGB"))
        mask = np.asarray(ex["label_mask"])

        if self.tfm:
            aug = self.tfm(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        if self.ignore_null:
            mask = mask.long()
            mask[mask == 255] = IGNORE_INDEX

        return {"image": img, "mask": mask}
