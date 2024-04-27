import torch
import numpy as np
from data_loader import LoadImaged
from add_channel_transform import AddChannelTransform
from libraries import*


def add_channel_dim(data):
    # Assuming the data is a NumPy array or a PyTorch tensor
    # Add a channel dimension at position 1 (the second position)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.unsqueeze(0)  # For single-channel data, e.g., grayscale images
    return data

train_transforms = Compose([
LoadImaged(keys=['img', 'seg']),  # Load image and segmentation label
AddChannelTransform(keys=['img', 'seg']),  # Apply the custom AddChannelTransform
ToTensord(keys=['img', 'seg'])  # Convert to PyTorch tensor if not already done
])
