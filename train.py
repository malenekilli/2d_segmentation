import math
from UNet import unet_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_prepros import process_all_files, generate_batches
from sklearn.model_selection import train_test_split
from losses import dice_loss




def train_model(base_path, batch_size, epochs):
    # Process all files to get the data
    data = process_all_files(base_path)

    # Split data into training plus validation and test sets
    train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Further split training plus validation set into actual training and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Calculate steps per epoch for training, validation and test
    train_steps = math.ceil(len(train_data) / batch_size)
    val_steps = math.ceil(len(val_data) / batch_size)
    
    # Create generators for training and validation
    train_generator = generate_batches(train_data, batch_size)
    val_generator = generate_batches(val_data, batch_size)
    
    # Load the UNet model with flexible input sizes
    model = unet_model(input_size=(None, None, 1))
    
    # Setup model saving mechanism and early stopping
    model_checkpoint = ModelCheckpoint('unet_best.keras', monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)

    # Fit the model using the generators
    model.fit(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    return model, test_data

if __name__ == "__main__":
    base_path = '/datasets/tdt4265/mic/asoca'
    model, test_data = train_model(base_path, batch_size=5, epochs=100)
    

