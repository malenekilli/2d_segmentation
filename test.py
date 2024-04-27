import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Test a simple computation
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1], [1, 1]]) 
print("Result of matrix multiplication: \n", tf.matmul(a, b))

# Test importing Keras modules
from tensorflow.keras.layers import Input, Conv2D
print("Successfully imported Input and Conv2D from tensorflow.keras.layers")
