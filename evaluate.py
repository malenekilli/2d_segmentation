import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import val_loader  # Assume there's a validation data loader
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# Ensure that GPU is being used if available
device_name = tf.test.gpu_device_name()
if device_name:
    print(f'Found GPU at: {device_name}')
else:
    print('GPU device not found, using CPU instead.')

# Load the trained model
model = load_model('model.h5')  # Update this to your model file path

# Metrics setup
dice_metric = DiceMetric(include_background=True, reduction="mean")
hd_metric = HausdorffDistanceMetric(percentile=95)

# Evaluation function to convert predictions and labels for metric computation
def evaluate_model(model, data_loader):
    dice_scores = []
    hausdorff_distances = []

    for batch in data_loader:
        images, true_masks = batch  # Adjust based on your data_loader output
        predicted_masks = model.predict(images)

        # Binarizing the output and true masks for Dice and HD95 calculation
        predicted_masks = (predicted_masks > 0.5).astype(np.int32)
        true_masks = (true_masks > 0.5).astype(np.int32)

        # MONAI expects torch tensors, so convert them
        predicted_masks = tf.convert_to_tensor(predicted_masks)
        true_masks = tf.convert_to_tensor(true_masks)

        # Update metrics
        dice_scores.append(dice_metric(predicted_masks, true_masks).numpy())
        hausdorff_distances.append(hd_metric(predicted_masks, true_masks).numpy())

    return np.mean(dice_scores), np.mean(hausdorff_distances)

# Running the evaluation
dice_score, hausdorff_distance = evaluate_model(model, val_loader)
print(f"Average Dice Score: {dice_score}")
print(f"Average Hausdorff Distance 95th Percentile: {hausdorff_distance}")
