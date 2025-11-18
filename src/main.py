from getData import getDataTrain1
from train import train
from eval import eval
import os
import tensorflow as tf
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



if __name__ == "__main__":
    # set experiment parameters
    max_length = 50
    label_pad = 14
    input_dim = 39
    num_classes = 29
    learning_rate = 0.0001 # Changed from 0.0001 to 0.00001. It's often good practice to lower the learning rate when resuming training
    batch_size = 64
    EPOCHS = 500 # Set to 1 to train for a single epoch after resuming from 200th epoch
    model_type = 'transformer' # 'rnn' or 'transformer'
    load_model = True
    monitor = 'val_loss'
    restore = False # Set to false when training from scratch
    drop_out = 0.2
    size = -1

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
        "transformer_tf_2_layers": {
            "model_type": "transformer_tf",
            "d_model": 128,
            "layer_size": 2, # num_layers
            "num_heads": 8,
            "d_ff": 512,
            "drop_out": 0.25 # Increased from 0.1. Experiment with values like 0.2, 0.25, or 0.3
        }
    }

    data = getDataTrain1(label_pad, max_length, num_classes)

    for exp_name, config in experiments.items():
        print(f"\n{'='*20} RUNNING EXPERIMENT: {exp_name} {'='*20}")
        SAVE_PATH = f"models/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{exp_name}/"
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        log = init_logging(SAVE_PATH)
        train(input_dim, num_classes, learning_rate, data, batch_size, size, EPOCHS, SAVE_PATH, restore, log, max_length, label_pad, **config)
        eval(SAVE_PATH, batch_size, log, label_pad, max_length, num_classes)