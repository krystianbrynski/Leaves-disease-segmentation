import os

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class LeafSegmentationDataset(Dataset):
    def __init__(self, image_dir, masks_dir, transform=None):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(masks_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        image_path = os.path.join(self.image_dir, self.images[id])
        mask_path = os.path.join(self.masks_dir, self.masks[id])

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


transform_train = A.Compose(
    [
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=360, p=0.7),
        ToTensorV2(),
    ],
)

transform_test = A.Compose(
    [
        A.Resize(512, 512),
        ToTensorV2(),
    ],
)


def datasets(train_images_dir: str, test_images_dir: str, train_masks_dir: str, test_masks_dir: str) -> tuple[LeafSegmentationDataset, LeafSegmentationDataset]:
    train_dataset = LeafSegmentationDataset(
        image_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=transform_train
    )

    test_dataset = LeafSegmentationDataset(
        image_dir=test_images_dir,
        masks_dir=test_masks_dir,
        transform=transform_test
    )

    return  train_dataset, test_dataset
