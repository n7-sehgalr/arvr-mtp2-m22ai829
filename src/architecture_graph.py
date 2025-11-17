import tensorflow as tf

# Load the Keras model saved earlier
loaded_keras_model = tf.saved_model.load("models/20251114-205711_rnn_2_layers")

loaded_keras_model.summary()