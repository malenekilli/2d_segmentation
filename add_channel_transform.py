from monai.transforms import Transform
from transform import add_channel_dim

class AddChannelTransform(Transform):
    """
    Add a channel dimension to the data at the specified position.
    This transform assumes the data is already a PyTorch tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = add_channel_dim(data[key])
        return data
