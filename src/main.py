from getData import getDataTrain1
from train import train
from eval import eval
import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.autograph.set_verbosity(0)

def init_logging(log_dir):
    logging_level = logging.INFO

    log_file = 'log_valid.txt'

    log_file = os.path.join(log_dir, log_file)
    if os.path.isfile(log_file):
        os.remove(log_file)

    # Get the root logger and remove all existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add a handler to write to the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('[[%(asctime)s]] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
    root_logger.addHandler(file_handler)

    # Add a handler to stream to the console
    root_logger.addHandler(logging.StreamHandler())
    root_logger.setLevel(logging_level)

    return logging



def run_experiments():
    # set experiment parameters
    max_length = 50
    label_pad = 14
    input_dim = 39
    num_classes = 29
    learning_rate = 0.01 # This is now the INITIAL learning rate for the schedule.
    batch_size = 64
    EPOCHS = 300 # Total epochs for the schedule
    model_type = 'transformer' # 'rnn' or 'transformer'
    load_model = True
    monitor = 'val_loss'
    restore = False # Set to false when training from scratch
    drop_out = 0.3
    size = -1

    # --- HParams Setup ---
    HP_MODEL_TYPE = hp.HParam('model_type', hp.Discrete(['rnn', 'transformer', 'transformer_tf']))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.00001, 0.001))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.5))
    HP_LAYER_SIZE = hp.HParam('layer_size', hp.IntInterval(1, 8))
    HP_D_MODEL = hp.HParam('d_model', hp.IntInterval(64, 512))

    METRIC_CER = 'final_cer_greedy'
    METRIC_WER = 'final_wer_greedy'

    hparams_log_dir = 'models/hparams'
    with tf.summary.create_file_writer(hparams_log_dir).as_default():
        hp.hparams_config(
            hparams=[HP_MODEL_TYPE, HP_LEARNING_RATE, HP_DROPOUT, HP_LAYER_SIZE, HP_D_MODEL],
            metrics=[hp.Metric(METRIC_CER, display_name='CER (Greedy)'), hp.Metric(METRIC_WER, display_name='WER (Greedy)')],
        )

    # --- Experiment Loop ---
    experiments = {
        # "rnn_2_layers": {
        #     "model_type": "rnn",
        #     "model_size": 400,
        #     "layer_size": 2,
        #     "drop_out": 0.2
        # },
        # "transformer_2_layers": {
        #     "model_type": "transformer",
        #     "model_size": 512,
        #     "layer_size": 2,
        #     "drop_out": 0.1 # Dropout is inside MultiHeadAttention for transformer
        # },
        # "transformer_6_layers": {
        #     "model_type": "transformer",
        #     "model_size": 512,
        #     "layer_size": 6,
        #     "drop_out": 0.1
        # },
        # "transformer_tf_1_layers": {
        #     "model_type": "transformer_tf",
        #     "d_model": 128,
        #     "layer_size": 1, # num_layers
        #     "num_heads": 2,
        #     "d_ff": 512,
        #     "drop_out": 0.25 # Increased from 0.1. Experiment with values like 0.2, 0.25, or 0.3
        # },
        "transformer_tf_1_layers_small": {
            "model_type": "transformer_tf",
            "d_model": 64,
            "layer_size": 1, # num_layers
            "num_heads": 2,
            "d_ff": 256,
            "drop_out": 0.25 # Increased from 0.1. Experiment with values like 0.2, 0.25, or 0.3
        }
    }

    data = getDataTrain1(label_pad, max_length, num_classes)

    for exp_name, config in experiments.items():
        print(f"\n{'='*20} RUNNING EXPERIMENT: {exp_name} {'='*20}")
        SAVE_PATH = f"models/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{exp_name}/"
        # SAVE_PATH = f"models/20251119-091023_transformer_tf_1_layers_small/"
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        log = init_logging(SAVE_PATH)

        # Combine base config with experiment-specific config for logging
        hparams = {
            'model_type': config.get('model_type', 'rnn'),
            'learning_rate': learning_rate,
            'dropout': config.get('drop_out', 0.0),
            'layer_size': config.get('layer_size', 0),
            'd_model': config.get('d_model', 0)
        }

        # Run training
        train(input_dim, num_classes, learning_rate, data, batch_size, size, EPOCHS, SAVE_PATH, restore, log, max_length, label_pad, **config)
        
        # Run evaluation
        final_metrics = eval(SAVE_PATH, batch_size, log, label_pad, max_length, num_classes)

        # Log HParams and final metrics
        run_dir = os.path.join(hparams_log_dir, os.path.basename(os.path.normpath(SAVE_PATH)))
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)
            tf.summary.scalar(METRIC_CER, final_metrics[0], step=1)
            tf.summary.scalar(METRIC_WER, final_metrics[1], step=1)
        log.info(f"Logged HParams and final metrics to {run_dir}")

if __name__ == "__main__":
    run_experiments()
