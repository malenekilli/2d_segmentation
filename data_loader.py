from data_prepros import train_data, val_data
from monai.data import DataLoader, Dataset


from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
    RandRotate90d,
    RandFlipd,
    ToTensord,
    RandAffine
)


def create_train_transforms():
    return Compose([
        ScaleIntensityRanged(
            keys=["img"],
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        RandAffine(
            prob=0.5,
            translate_range=(15,15),
            scale_range=(0.1,0.1),
            rotate_range=(0.1)
        ),
        # RandFlipd(
        #     keys=["img", "seg"],
        #     prob=0.5,
        #     spatial_axis=[1,2]
        # ),
        # RandRotate90d(
        #     keys=["img", "seg"],
        #     prob=1.0,
        #     spatial_axes=[1,2]
        # ),
        ToTensord(keys=["img", "seg"])
        
    ])

def create_val_transforms():
    return Compose([
        ScaleIntensityRanged(
            keys=["img"],
            a_min=0, a_max=255,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=["img", "seg"])
    ])

# Example usage of the transform
train_transforms = create_train_transforms()
val_transforms = create_val_transforms()


# # Transform the data using training and validation transforms
# transformed_train_data = [train_transforms(item) for item in train_data]
# transformed_val_data = [val_transforms(item) for item in val_data]

# print("Transformed Train Image Shape:", transformed_train_data["img"].shape)
# print("Transformed Val Image Shape:", transformed_val_data["seg"].shape)

    
    
train_dataset=Dataset(train_data, train_transforms)
val_dataset= Dataset(val_data, val_transforms)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader=DataLoader(val_dataset,batch_size=2, shuffle=True, num_workers=4)


# # Iterate through one epoch to see the debug output
# for i, data in enumerate(train_loader):
#     pass  # The debug statements will print during iteration

# for i, data in enumerate(val_loader):
#     pass  # Debug for validation data



import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class CustomMedicalDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list of dicts): Data where each entry is a dict with keys 'img' and 'seg'.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if sample['img'].ndim == 2:
            sample['img'] = sample['img'][np.newaxis, ...]  # (1, H, W)
        if sample['seg'].ndim == 2:
            sample['seg'] = sample['seg'][np.newaxis, ...]  # (1, H, W)

        # Debug before transformation
        #print(f"Before transform - Image shape: {sample['img'].shape}, Segmentation shape: {sample['seg'].shape}")

        if self.transform:
            sample = self.transform(sample)

        # Debug after transformation
        #print(f"After transform - Image shape: {sample['img'].shape}, Segmentation shape: {sample['seg'].shape}")

        
        return sample

train_dataset = CustomMedicalDataset(train_data, transform=create_train_transforms())
val_dataset = CustomMedicalDataset(val_data, transform=create_val_transforms())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

# Example of how to use the data loader
    
# try:
#     for i, data in enumerate(train_loader):
#         print(f"Batch {i} - Image shapes: {data['img'].shape}, Segmentation shapes: {data['seg'].shape}")
# except Exception as e:
#     print(f"Error processing batch {i}: {str(e)}")
#     raise
