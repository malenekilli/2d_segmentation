import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from UNet import model
from UNet import unet_model

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6  # A small constant to avoid division by zero, more numerically stable
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


model.compile(optimizer=Adam(learning_rate=1e-5), loss=combined_loss, metrics=['accuracy'])

model = UNet(dimensions=2, in_channels=1, out_channels=1, channels=(64, 128, 256, 512), strides=(2, 2, 2))
loss_function = DiceLoss()