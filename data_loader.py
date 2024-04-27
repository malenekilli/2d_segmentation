from libraries import *
from data_paths import train_files

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    RandRotate90d,
    ToTensord
)


# Keys in the dictionary
keys = ["img", "seg"]  # 'img' for image, 'seg' for segmentation masks if applicable

train_transforms = Compose([
    LoadImaged(keys=keys),
    ScaleIntensityRanged(
        keys=["img"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True
    ),
    RandRotate90d(keys=keys, prob=0.5, spatial_axes=[0, 1]),
    ToTensord(keys=keys)
])


train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
