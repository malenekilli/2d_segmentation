import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_prepros import process_all_files, generate_batches
from train import train_model
from losses import dice_coefficient, dice_loss, combined_loss  # Import the functions

def plot_images(true_images, pred_images, num_images=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(true_images[i].squeeze(), cmap='gray')
        ax.title.set_text('Ground Truth')
        plt.axis('off')
        ax = plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(pred_images[i].squeeze(), cmap='gray')
        ax.title.set_text('Prediction')
        plt.axis('off')
    plt.show()

def evaluate_model(model_path, test_data, batch_size=5, threshold=0.5):
    model = load_model(model_path, custom_objects={'combined_loss': combined_loss, 'accuracy': 'accuracy'})  # Add custom_objects if needed
    test_batches = generate_batches(test_data, batch_size)
    test_steps = len(test_data) // batch_size
    dice_scores = []
    # total_loss = 0
    # total_accuracy = 0

    for i in range(test_steps):
        X_test, y_test = next(test_batches)
        # results = model.evaluate(X_test, y_test, verbose=1)
        # total_loss += results[0]
        # total_accuracy += results[1]

        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > threshold).astype(np.int32)
        dice_score = dice_coefficient(y_test, y_pred_binary)
        dice_scores.append(dice_score)
    
    average_dice_score = np.mean(dice_scores)
    print("Threshold:", threshold, "Average Dice Coefficient:", average_dice_score)

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for th in thresholds:
        evaluate_model(model_path, test_data, batch_size=5, threshold=th)

        # if i == 0 or i == 5:
        #     plot_images(y_test, y_pred_binary, num_images=5)

    # average_loss = total_loss / test_steps
    # average_accuracy = total_accuracy / test_steps
    # average_dice_score = np.mean(dice_scores)

    # print("Average Loss:", average_loss)
    # print("Average Accuracy:", average_accuracy)
    # print("Average Dice Coefficient:", average_dice_score)

if __name__ == "__main__":
    base_path = '/datasets/tdt4265/mic/asoca'
    data = process_all_files(base_path)
    model_path = '/work/malenelk/project_2D/unet_best.keras'
    _, test_data = train_model(base_path, batch_size=5, epochs=100)
    evaluate_model(model_path, test_data, batch_size=5)
