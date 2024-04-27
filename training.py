from libraries import *
from UNet import model,device
from data_loader import train_loader
from monai.transforms import Compose, LoadImaged, ToTensord
from add_channel_transform import AddChannelTransform
from transform import train_transforms

loss_function = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

max_epochs = 100
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['img'].to(device), batch_data['seg'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

train_files = [
    {'img': 'path/to/image1.nii', 'seg': 'path/to/seg1.nii'},
    {'img': 'path/to/image2.nii', 'seg': 'path/to/seg2.nii'}
]

# Create dataset and dataloader
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

print(model)
print(device)