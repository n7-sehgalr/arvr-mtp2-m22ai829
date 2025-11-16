import tensorflow as tf
from tensorflow import keras
import numpy as np

class SequenceAccuracy(keras.metrics.Metric):
    def __init__(self, name='sequence_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, seq_len_list,label_data_length, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        # logit_length = tf.fill([batch_size], y_pred_shape[1])
        logit_length = seq_len_list

        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        
        # Convert sparse decoded tensor to dense and ensure it's a TF tensor
        y_pred_decoded = tf.sparse.to_dense(decoded[0], default_value=0)
        y_pred_decoded = tf.cast(y_pred_decoded, y_true.dtype)

        # Pad the shorter tensor to match the length of the longer one
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred_decoded)
        max_len = tf.maximum(y_true_shape[1], y_pred_shape[1])
        y_true_padded = tf.pad(y_true, [[0, 0], [0, max_len - y_true_shape[1]]])
        y_pred_padded = tf.pad(y_pred_decoded, [[0, 0], [0, max_len - y_pred_shape[1]]])

        num_errors = tf.math.reduce_any(
            tf.math.not_equal(y_true_padded, y_pred_padded), axis=1)
        num_errors = tf.cast(num_errors, tf.float32)
        num_errors = tf.reduce_sum(num_errors)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)

    def result(self):
        return self.count / self.total

    def to_dense(self, tensor):
        tensor = tf.sparse.to_dense(tensor, default_value=0)
        return tensor

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)


class EditDistance(keras.metrics.Metric):
    def __init__(self, name='edit_distance', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.sum_distance = self.add_weight(name='sum_distance', 
                                            initializer='zeros')
                
    def update_state(self, y_true, y_pred,  seq_len_list,label_data_length, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        logit_length = seq_len_list
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        sum_distance = tf.math.reduce_sum(tf.edit_distance(tf.cast(decoded[0],tf.int64), tf.cast(tf.sparse.from_dense(
    y_true),tf.int64)))
        batch_size = tf.cast(batch_size, tf.float32)
        self.sum_distance.assign_add(sum_distance)
        self.total.assign_add(batch_size)

    def result(self):
        return self.sum_distance / self.total

    def reset_states(self):
        self.sum_distance.assign(0)
        self.total.assign(0)