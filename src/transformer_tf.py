import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        
        # Based on the formula from "Attention Is All You Need"
        pos = np.arange(max_seq_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        """
        Args:
            x: A tensor of shape (batch_size, sequence_length, d_model)
        Returns:
            A tensor of shape (batch_size, sequence_length, d_model)
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

class FeedForwardSubLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super(FeedForwardSubLayer, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.fc2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask=None):
        attn_output = self.self_attn(query=x, value=x, key=x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)
        
        ff_output = self.ff_sublayer(out1)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.norm2(out1 + ff_output)
        
        return out2

class TransformerEncoder(tf.keras.Model):
    def __init__(self, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.d_model = d_model
        # This Dense layer projects the input features into the model's dimension (d_model)
        self.input_projection = tf.keras.layers.Dense(d_model)
        self.positional_encoding = PositionalEncoding(max_seq_length, d_model)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask=None):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)
        
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, training=training, mask=mask)
            
        return x

class RegressionHead(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(RegressionHead, self).__init__(**kwargs)
        self.fc = tf.keras.layers.Dense(output_dim)
    
    def call(self, x):
        return self.fc(x)

def create_transformer_model(input_shape, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length, num_classes):
    """
    Factory function to create the full Transformer model with a regression head.
    """
    # The input is now a sequence of feature vectors, not integer token IDs.
    inputs = tf.keras.Input(shape=input_shape)
    # Masking is not implemented here as the input is dense. It can be added if needed.

    encoder = TransformerEncoder(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length
    )
    
    encoder_output = encoder(inputs, training=True, mask=None)
    
    regression_head = RegressionHead(output_dim=num_classes)
    
    outputs = regression_head(encoder_output)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
