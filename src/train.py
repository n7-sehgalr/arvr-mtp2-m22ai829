import tensorflow as tf
from model import rnn_att_model, CTCLoss
from transformer_model import build_transformer_model
from tensorflow import keras
from transformer_tf import create_transformer_model
from metrics import EditDistance, SequenceAccuracy
from getData import tfdata1, getDataTest
import numpy as np
import os
import time
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

letter_table = ['','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','','']
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]/max(len(s1), len(s2))


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate schedule that decays in a piecewise linear fashion.
    - Epochs 0-100: Decays linearly from initial_lr (e.g., 0.01) to mid_lr (e.g., 0.001).
    - Epochs 101-300: Decays linearly from mid_lr to final_lr (e.g., 0.0001).
    - Epochs > 300: Stays at final_lr.
    """
    def __init__(self, initial_learning_rate, mid_learning_rate, final_learning_rate, decay_steps_1, decay_steps_2):
        super(CustomLearningRateSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.mid_learning_rate = mid_learning_rate
        self.final_learning_rate = final_learning_rate
        self.decay_steps_1 = decay_steps_1
        self.decay_steps_2 = decay_steps_2
        self.total_decay_steps = decay_steps_1 + decay_steps_2

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # First decay phase
        lr_decay_1 = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps_1,
            end_learning_rate=self.mid_learning_rate,
            power=1.0 # Linear decay
        )
        
        # Second decay phase
        lr_decay_2 = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.mid_learning_rate,
            decay_steps=self.decay_steps_2,
            end_learning_rate=self.final_learning_rate,
            power=1.0 # Linear decay
        )

        return tf.cond(
            step < self.decay_steps_1,
            lambda: lr_decay_1(step),
            lambda: tf.cond(
                step < self.total_decay_steps,
                lambda: lr_decay_2(step - self.decay_steps_1),
                lambda: self.final_learning_rate
            )
        )

class EpochAwareLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A wrapper that makes a step-based learning rate schedule epoch-based.
    """
    def __init__(self, schedule, steps_per_epoch):
        super().__init__()
        self.schedule = schedule
        self.steps_per_epoch = tf.cast(steps_per_epoch, tf.float32)

    def __call__(self, step):
        # Convert the global step to an epoch number
        epoch = step / self.steps_per_epoch
        return self.schedule(epoch)


# @tf.function
def to_dense(tensor):
    tensor = tf.sparse.to_dense(tensor, default_value=0)
    tensor = tf.cast(tensor, tf.int32).numpy()
    return tensor


# @tf.function
def loss(model, x, y,seq_len_list,label_data_length, training):
  y_ = model(x, training=training)
  loss_object = CTCLoss()

  return loss_object(y_true = y,y_pred = y_,data_sequence = seq_len_list,label_seq = label_data_length)

# @tf.function
def grad(model, inputs, targets,seq_len_list,label_data_length):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, seq_len_list,label_data_length,training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

# @tf.function
def train_step(epoch, train1Data,model, optimizer,epoch_loss_avg, epoch_edit_dist, epoch_seq_acc, log, train_summary_writer):
    # log.info('---------------------------- Start Train Epoch {} ---------------------------------\n'.format(epoch))
    for step, (x_batch_train, y_batch_train,seq_len_list,label_data_length) in enumerate(train1Data):
        loss_value, grads = grad(model, x_batch_train, y_batch_train,seq_len_list,label_data_length)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss_avg.update_state(loss_value)
        y_pred = model(x_batch_train, training=True)
        epoch_edit_dist.update_state(y_batch_train, y_pred,seq_len_list,label_data_length)
        epoch_seq_acc.update_state(y_batch_train, y_pred,seq_len_list,label_data_length)

        if step % 50 == 0:
            log.info("Step {:03d}: Loss: {:.3f}, EditDistance: {:.3%}".format(step,
                                                                             epoch_loss_avg.result(),
                                                                             epoch_edit_dist.result()))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_loss_avg.result(), step=optimizer.iterations)
                tf.summary.scalar('edit_distance', epoch_edit_dist.result(), step=optimizer.iterations)
                tf.summary.scalar('sequence_accuracy', epoch_seq_acc.result(), step=optimizer.iterations)
    log.info('---------------------------- Finish Train Epoch {} ---------------------------------\n'.format(epoch))


def test(batch_size,model, log,label_pad, max_length, num_classes, summary_writer, step):
    input_data, label_data,seq_len_list,label_data_length = getDataTest(batch_size,label_pad, max_length, num_classes)
    start_time = time.time()
    val_logits = model(input_data)
    inference_time = time.time() - start_time
    for _ in range(10):
        i = int(np.random.randint(0,len(label_data)))
        target = ''.join(
            [letter_table[x] for x in label_data[i]])

        sequence_length = seq_len_list

        decoded_gd, _ = tf.nn.ctc_greedy_decoder(
            tf.transpose(val_logits, perm=[1, 0, 2]), sequence_length, merge_repeated=True
        )
        dense_decoded = to_dense(decoded_gd[0])
        predict_gd = ''.join(
            [letter_table[x] for x in dense_decoded[i]])

        decoded_bs, _ = tf.nn.ctc_beam_search_decoder(
            tf.transpose(val_logits, perm=[1, 0, 2]), sequence_length, beam_width=10)
        dense_decoded = to_dense(decoded_bs[0])
        predict_bs = ''.join(
            [letter_table[x] for x in dense_decoded[i]])

        dist = levenshteinDistance(target, predict_gd)

        log.info('Target                : "{}"'.format(target))
        log.info('Predict Greedy Search : "{}"'.format(predict_gd))
        log.info('Levenshtein Distance  :  {}\n'.format(dist))
        log.info('Predict Beam Search   : "{}"\n'.format(predict_bs))
    
    with summary_writer.as_default():
        tf.summary.scalar('inference_time_seconds', inference_time, step=step)

# @tf.function
def validation(valid1Data, model, val_edit_dist, val_seq_acc, val_loss_avg, log, start_time):
    for x_batch_val, y_batch_val, seq_len_list,label_data_length in valid1Data:
        val_logits = model(x_batch_val, training=False)
        val_loss = loss(model, x_batch_val, y_batch_val, seq_len_list, label_data_length, training=False)
        # Update val metrics
        val_loss_avg.update_state(val_loss)
        val_edit_dist.update_state(y_batch_val, val_logits, seq_len_list,label_data_length)
        val_seq_acc.update_state(y_batch_val, val_logits, seq_len_list, label_data_length)

    val_ed = val_edit_dist.result()
    val_sa = val_seq_acc.result()
    val_l = val_loss_avg.result()

    val_edit_dist.reset_state()
    val_seq_acc.reset_state()
    val_loss_avg.reset_state()

    log.info("Validation Loss: %.4f, Validation EditDistance: %.4f, Validation SequenceAccuracy: %.4f" % (float(val_l), float(val_ed), float(val_sa)))
    return val_l, val_ed, val_sa

    # log.info("Time taken: %.2fs\n" % (time.time() - start_time))

def checkpoint_save(log, checkpoint, manager):
    checkpoint.step.assign_add(1)
    if int(checkpoint.step) % 1 == 0:
        save_path = manager.save()
        log.info("Saved checkpoint for ckpt step {}: {}".format(int(checkpoint.step), save_path))

def log_model_summary(model, summary_writer):
    with summary_writer.as_default():
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        tf.summary.text('model_summary', '\n'.join(string_list), step=0)

def train(input_shape2, num_classes, learning_rate,data,
          batch_size, size, EPOCHS, SAVE_PATH,
          restore,log,max_length,label_pad, model_type, optimizer_name='adam', log_histograms=False,
          **kwargs):

    train1Data,valid1Data = tfdata1(data, batch_size, size)

    number_train = len(train1Data)
    number_valid = len(valid1Data)

    if model_type == 'transformer':
        log.info(f"Using Transformer model with {kwargs['layer_size']} layers.")
        model = build_transformer_model(input_shape2=input_shape2, output_shape=num_classes, max_length=max_length, model_size=kwargs['model_size'], num_layers=kwargs['layer_size'])
    elif model_type == 'transformer_tf':
        log.info(f"Using transformer_tf model with {kwargs['layer_size']} layers.")
        model = create_transformer_model(
            input_shape=(None, input_shape2),
            d_model=kwargs['d_model'],
            num_layers=kwargs['layer_size'],
            num_heads=kwargs['num_heads'],
            d_ff=kwargs['d_ff'],
            dropout=kwargs['drop_out'],
            max_seq_length=max_length,
            num_classes=num_classes
        )
    else: # default to rnn
        log.info(f"Using RNN model with {kwargs['layer_size']} layers.")
        model = rnn_att_model(input_shape2 = input_shape2, output_shape = num_classes, dropout = kwargs['drop_out'],num_units = kwargs['model_size'],num_layers=kwargs['layer_size'])


    log.info('number of train samples {}'.format(number_train))
    log.info('number of valid samples {}'.format(number_valid))

    # purpose: save and restore models
    checkpoint_path = SAVE_PATH +"training_checkpoints/"
    
    # TensorBoard setup
    train_log_dir = SAVE_PATH + 'logs/train'
    val_log_dir = SAVE_PATH + 'logs/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    log_model_summary(model, train_summary_writer)

    model.summary()
    
    # Calculate steps per epoch to make the LR schedule epoch-aware
    steps_per_epoch = number_train

    # --- Setup Optimizer ---
    # If starting a new run, use the learning rate schedule.
    # If resuming, we will use the fixed learning rate passed from main.py.
    if not restore:
        # For a new run, just use the fixed learning rate for simplicity as requested.
        # The schedule can be re-enabled here if needed.
        lr_to_use = learning_rate
        # step_based_schedule = CustomLearningRateSchedule(
        #     initial_learning_rate=learning_rate,
        #     mid_learning_rate=0.001,
        #     final_learning_rate=0.0001,
        #     decay_steps_1=100,
        #     decay_steps_2=200)
        # lr_to_use = EpochAwareLearningRateSchedule(step_based_schedule, steps_per_epoch)
    else:
        lr_to_use = learning_rate # Use the fixed value for fine-tuning

    if optimizer_name.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr_to_use)
    elif optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr_to_use, beta_2=0.98)
    elif optimizer_name.lower() == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=lr_to_use, beta_2=0.98)
    else: # Default to Adam
        optimizer = keras.optimizers.Adam(learning_rate=lr_to_use)

    checkpoint = tf.train.Checkpoint(step = tf.Variable(1),
                                     model = model,
                                     generator_optimizer=optimizer)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        log.info('check point not exist {}'.format(checkpoint_path))
    else:
        log.info('check point {}'.format(checkpoint_path))
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=5)

    initial_epoch = 0
    if restore and manager.latest_checkpoint:
        # Use expect_partial to avoid warnings if the optimizer state doesn't match
        # Removing .expect_partial() makes the restoration strict.
        # This will raise an error if the optimizer state cannot be restored,
        # which is what we want if the architectures don't match. We use expect_partial()
        # when we intentionally change architecture and only want to restore model weights.
        status = checkpoint.restore(manager.latest_checkpoint).expect_partial()
        # status.assert_consumed() # This would raise an error if the optimizer state doesn't match.
        initial_epoch = int(checkpoint.step)
        log.info(f"Restored from {manager.latest_checkpoint}. Resuming training from epoch {initial_epoch}.")

    #------------------ inspection train mode ------------------
    train_loss_results = []
    train_accuracy_results = []
    log.info('Save Path : {}'.format(SAVE_PATH))
    # The loop now starts from the restored epoch
    for epoch in range(initial_epoch, initial_epoch + EPOCHS):

        start_time = time.time()

        # Metrics
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_edit_dist = EditDistance()
        epoch_seq_acc = SequenceAccuracy()
        
        val_loss_avg = tf.keras.metrics.Mean()
        val_edit_dist = EditDistance()
        val_seq_acc = SequenceAccuracy()

        train_step(epoch, train1Data,model, optimizer,epoch_loss_avg, epoch_edit_dist, epoch_seq_acc, log, train_summary_writer)
        
        epoch_train_time = time.time() - start_time

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_edit_dist.result())

        # Get current learning rate, handling both schedule and fixed value cases
        if isinstance(optimizer.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.learning_rate(epoch)
        else:
            current_lr = optimizer.learning_rate

        # Get the float value, whether current_lr is a tensor or already a float
        lr_value = current_lr.numpy() if hasattr(current_lr, 'numpy') else current_lr

        log.info("Epoch {:03d}: Loss: {:.3f}, EditDistance: {:.3%}, LR: {:.6f}".format(epoch, epoch_loss_avg.result(), epoch_edit_dist.result(), lr_value))
        
        # Log training metrics for the epoch
        with train_summary_writer.as_default():
            tf.summary.scalar('learning_rate', current_lr, step=epoch)
            tf.summary.scalar('epoch_loss', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('epoch_edit_distance', epoch_edit_dist.result(), step=epoch)
            tf.summary.scalar('epoch_sequence_accuracy', epoch_seq_acc.result(), step=epoch)
            tf.summary.scalar('epoch_training_time_seconds', epoch_train_time, step=epoch)
            # Log weights (this is very I/O intensive and slows down training)
            if log_histograms:
                for layer in model.layers:
                    for weight in layer.weights:
                        tf.summary.histogram(f'{layer.name}/{weight.name}', weight, step=epoch)

        epoch_loss_avg.reset_state()
        epoch_edit_dist.reset_state()
        epoch_seq_acc.reset_state()

        checkpoint_save(log, checkpoint, manager)
        val_loss, val_ed, val_sa = validation(valid1Data, model, val_edit_dist, val_seq_acc, val_loss_avg, log, start_time)
        with val_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', val_loss, step=epoch)
            tf.summary.scalar('epoch_edit_distance', val_ed, step=epoch)
            tf.summary.scalar('epoch_sequence_accuracy', val_sa, step=epoch)

        log.info("Time taken: %.2fs\n" % (time.time() - start_time))
        if epoch%20 == 0:
            test(batch_size, model, log,label_pad, max_length, num_classes, val_summary_writer, epoch)

    # Save the model in the modern Keras v3 format.
    model_save_path = os.path.join(SAVE_PATH, "model.keras")
    model.save(model_save_path)
    log.info(f"Model saved in Keras format to: {model_save_path}")
