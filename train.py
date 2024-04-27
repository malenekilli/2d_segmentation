import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from data_loader import train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

# Loss and optimizer
loss_function = DiceLoss(to_onehot_y=False, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Data loaders
base_path = '/datasets/tdt4265/mic/asoca'

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, masks = batch['img'].to(device), batch['seg'].to(device)

        print(f'Debug: Image Channels = {images.shape[1]}, Mask Channels = {masks.shape[1]}')


        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')  # Fixed: Added closing parenthesis here
