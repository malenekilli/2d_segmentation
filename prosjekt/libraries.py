import os
import monai
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Resized,
    ToTensord
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.optimizers import Novograd

import torch
print_config()

