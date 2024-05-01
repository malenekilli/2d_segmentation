import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from data_loader import train_loader, val_loader
from monai.networks.layers import Norm
from torch.nn.functional import sigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Model setup
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(4, 8, 16, 32, 64),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout=0.01,
    norm=Norm.INSTANCE
).to(device)

# Loss and optimizer
loss_function = DiceCELoss(to_onehot_y=False, softmax=False).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)

num_epochs = 30
# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Data loaders
base_path = '/datasets/tdt4265/mic/asoca'

best_val_loss = float('inf')
best_model_path = "model.pth"

# Training loop

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        images, masks = batch['img'].to(device), batch['seg'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        outputs = sigmoid(outputs)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_images, val_masks = val_batch['img'].to(device), val_batch['seg'].to(device)
            val_outputs = model(val_images)
            val_outputs = sigmoid(outputs)
            val_loss += loss_function(val_outputs, val_masks).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved new best model with validation loss: {best_val_loss}')

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR: {current_lr}')

# Plotting
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
