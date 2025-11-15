import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.dense_input = layers.Dense(output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

        position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, output_dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / output_dim))
        
        self.positional_encoding = tf.Variable(tf.zeros((sequence_length, output_dim)), trainable=False)
        
        self.positional_encoding[:, 0::2].assign(tf.sin(position * div_term))

        self.positional_encoding[:, 1::2].assign(tf.cos(position * div_term))

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        embedded_positions = self.positional_encoding[:length, :]
        embedded_inputs = self.dense_input(inputs)
        return embedded_inputs + embedded_positions


def build_transformer_model(input_shape2, output_shape, max_length, model_size=256, num_heads=4, dense_dim=256, num_layers=2):
    input_shape = (None, input_shape2)
    inputs = keras.Input(shape=input_shape)

    # Positional Embedding
    x = PositionalEmbedding(
        sequence_length=max_length, input_dim=input_shape2, output_dim=model_size
    )(inputs)

    # Transformer Encoder blocks
    for _ in range(num_layers):
        x = TransformerEncoder(embed_dim=model_size, dense_dim=dense_dim, num_heads=num_heads)(x)

    # Output layer
    outputs = layers.Dense(output_shape)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
