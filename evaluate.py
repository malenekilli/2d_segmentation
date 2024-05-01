import torch
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from data_loader import val_loader
from monai.networks.nets import UNet
from monai.networks.layers import Norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
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

model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set model to evaluation mode

# Metrics setup
dice_metric = DiceMetric(include_background=True, reduction="mean")
hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

# Function to compute metrics and visualize results
def evaluate_model(model, data_loader, num_samples=10):
    dice_scores = []
    hd_scores = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images, labels = batch['img'].to(device), batch['seg'].to(device)
            outputs = torch.sigmoid(model(images))
            predictions = (outputs > 0.5).float()

            if torch.sum(predictions) == 0 and torch.sum(labels) == 0:
                # Both prediction and label are empty, perfect match
                dice_score = 1.0
                hd_score = 0.0  # No distance between empty sets
            else:
                dice_metric.reset()
                dice_metric(y_pred=predictions, y=labels)
                dice_score = dice_metric.aggregate().item()
                
                hd_metric.reset()
                hd_metric(y_pred=predictions, y=labels)
                hd_score = hd_metric.aggregate().item()

            dice_scores.append(dice_score)
            hd_scores.append(hd_score)

            if i < num_samples:
                visualize_segmentation(images[0], predictions[0], labels[0], dice_score, hd_score)

        return dice_scores, hd_scores


# Visualization function for segmentation results
def visualize_segmentation(image, prediction, label, dice_score, hd_score):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.cpu().squeeze(), cmap='gray')  # ensure image is 2D
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(prediction.cpu().squeeze(), cmap='gray')  # ensure prediction is 2D
    plt.title(f'Predicted Mask\nDice: {dice_score:.2f}, HD95: {hd_score:.2f}')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(label.cpu().squeeze(), cmap='gray')  # ensure label is 2D
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.show()

# Evaluate the model
dice_scores, hd_scores = evaluate_model(model, val_loader, num_samples=5)
print(f'Average Dice Score: {sum(dice_scores) / len(dice_scores):.4f}')
print(f'Average HD95 Score: {sum(hd_scores) / len(hd_scores):.4f}')
