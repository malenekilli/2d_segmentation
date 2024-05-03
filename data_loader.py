from data_prepros import train_data, val_data, test_data
from monai.data import DataLoader, Dataset
from monai.transforms import MapTransform
from monai.transforms import(
    Compose,
    ScaleIntensityRanged,
    ToTensord,
    RandAffine,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandAdjustContrast,
    Rand2DElastic,
    RandFlip,
    RandRotate90,
    EnsureType

    )
class RandElasticDeform2DWithKeys(MapTransform):
    def __init__(self, keys, prob=0.2, spacing=(30, 30), magnitude_range=(5, 10)):
        super().__init__(keys)
        self.elastic_deform = Rand2DElastic(
            prob=prob,
            spacing=spacing,
            magnitude_range=magnitude_range,
            padding_mode='border'
        )

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.elastic_deform(data[key])
        return data
class RandFlipWithKeys(MapTransform):
    def __init__(self, keys, prob=0.5, spatial_axis=0):
        super().__init__(keys)
        self.flip = RandFlip(prob=prob, spatial_axis=spatial_axis)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.flip(data[key])
        return data

class RandRotate90WithKeys(MapTransform):
    def __init__(self, keys, prob=0.5, max_k=3):
        super().__init__(keys)
        self.rotate = RandRotate90(prob=prob, max_k=max_k)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.rotate(data[key])
        return data

class RandAffineWithKeys(MapTransform):
    def __init__(self, keys, prob=0.5, translate_range=(15, 15), scale_range=(0.1, 0.1), rotate_range=(0.1)):
        super().__init__(keys)
        self.rand_affine = RandAffine(prob=prob, translate_range=translate_range, scale_range=scale_range, rotate_range=rotate_range)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.rand_affine(data[key])
        return data

class RandGaussianNoiseWithKeys(MapTransform):
    def __init__(self, keys, prob=0.1, mean=0.0, std=0.1):
        super().__init__(keys)
        self.transform = RandGaussianNoise(prob=prob, mean=mean, std=std)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
        return data

class RandGaussianSmoothWithKeys(MapTransform):
    def __init__(self, keys, prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)):
        super().__init__(keys)
        self.transform = RandGaussianSmooth(prob=prob, sigma_x=sigma_x, sigma_y=sigma_y)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
        return data

class RandAdjustContrastWithKeys(MapTransform):
    def __init__(self, keys, prob=0.3, gamma=(0.5, 1.5)):
        super().__init__(keys)
        self.transform = RandAdjustContrast(prob=prob, gamma=gamma)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.transform(data[key])
        return data

def create_train_transforms():
    return Compose([
        ScaleIntensityRanged(
             keys=["img"],
             a_min=0, a_max=255, 
             b_min=0.0, b_max=1.0,
             clip=True
         ),

         RandElasticDeform2DWithKeys(
            keys=["img", "seg"],
            prob=0.2,
            spacing=(30, 30),
            magnitude_range=(5, 10)
        ),
        
        RandAffineWithKeys(
            keys=["img", "seg"],
            prob=0.5,
            translate_range=(15, 15),
            scale_range=(0.1, 0.1),
            rotate_range=(0.1)
        ),
        
        RandGaussianNoiseWithKeys(
           keys=["img"],
           prob=0.3,
           std=0.1
        ),
    #
        RandGaussianSmoothWithKeys(
           keys=["img"],
           prob=0.2,
           sigma_x=(0.5, 1.5),
           sigma_y=(0.5, 1.5)
        ),
        RandAdjustContrastWithKeys(
           keys=["img"],
           prob=0.3,
           gamma=(0.5, 1.5)
        ),
        RandFlipWithKeys(
            keys=["img", "seg"],
            prob=0.5,
            spatial_axis=1  # Flip horizontally, change to 0 for vertical
        ),
        RandRotate90WithKeys(
            keys=["img", "seg"],
            prob=0.5,
            max_k=3  # Rotate by 0, 90, 180, 270 degrees
        ),
        EnsureType(), 

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
test_transforms=create_val_transforms()


    
    
train_dataset=Dataset(train_data, train_transforms)
val_dataset= Dataset(val_data, val_transforms)
test_dataset=Dataset(test_data,test_transforms)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader=DataLoader(val_dataset,batch_size=2, shuffle=True, num_workers=4)
test_loader=DataLoader(test_dataset, batch_size=2,shuffle=True,num_workers=4)




import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class CustomMedicalDataset(Dataset):
    def __init__(self, data, transform=None):
    #     """
        # Args:
        #     data (list of dicts): Data where each entry is a dict with keys 'img' and 'seg'.
        #     transform (callable, optional): Transform to be applied on a sample.
        # """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]

        img = sample['img']
        seg = sample['seg']

        # Ensure data has at least a channel dimension
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # Adding channel dimension
        if seg.ndim == 2:
            seg = seg[np.newaxis, ...]  # Adding channel dimension

        sample['img'] = img
        sample['seg'] = seg

        if self.transform:
            sample = self.transform(sample)

        return sample

    

   

train_dataset = CustomMedicalDataset(train_data, transform=create_train_transforms())
val_dataset = CustomMedicalDataset(val_data, transform=create_val_transforms())
test_dataset=CustomMedicalDataset(test_dataset,transform=create_val_transforms())

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
test_loader=DataLoader(test_dataset, batch_size=2, shuffle=False,num_workers=4,)


