import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(size: tuple[int, int] = (224, 224)) -> tuple[A.Compose, A.Compose]:
    train_tfms = A.Compose(
        [
            A.SmallestMaxSize(max_size=int(size[0] * 1.5)),
            A.RandomCrop(*size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    val_tfms = A.Compose(
        [
            A.SmallestMaxSize(max_size=size[0]),
            A.CenterCrop(*size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return train_tfms, val_tfms
