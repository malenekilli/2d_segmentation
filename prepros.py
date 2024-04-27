from libraries import*

train_transforms = Compose([
    LoadImaged(keys=['img', 'seg']),
    ScaleIntensityRanged(keys=['img'], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=['img', 'seg'], spatial_size=(128, 128, 64)),
    ToTensord(keys=['img', 'seg'])
])
