from getData import getDataTrain1
from train import train
from eval import eval
import os
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
import logging
import random
import itertools


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.autograph.set_verbosity(0)

# --- HParams and Logging Setup (run once) ---
HP_MODEL_TYPE = hp.HParam('model_type', hp.Discrete(['rnn', 'transformer', 'transformer_tf']))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.00001, 0.001))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.5))
HP_LAYER_SIZE = hp.HParam('layer_size', hp.IntInterval(1, 8))
HP_D_MODEL = hp.HParam('d_model', hp.IntInterval(64, 512))
HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([1, 2, 4, 8]))
HP_D_FF = hp.HParam('d_ff', hp.Discrete([128, 256, 512]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['rmsprop', 'adam', 'adamw']))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64, 128]))

METRIC_CER = 'final_cer_greedy'
METRIC_WER = 'final_wer_greedy'

hparams_log_dir = 'models/hparams'
with tf.summary.create_file_writer(hparams_log_dir).as_default():
    hp.hparams_config(
        hparams=[HP_MODEL_TYPE, HP_LEARNING_RATE, HP_DROPOUT, HP_LAYER_SIZE, HP_D_MODEL, HP_NUM_HEADS, HP_D_FF,
                 HP_OPTIMIZER, HP_BATCH_SIZE],
        metrics=[hp.Metric(METRIC_CER, display_name='CER (Greedy)'), hp.Metric(METRIC_WER, display_name='WER (Greedy)')],
    )

def setup_logging():
    """Sets up the root logger with a console handler. This should be called only once."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid adding handlers if they already exist
    if not root.handlers:
        root.addHandler(logging.StreamHandler())

setup_logging()
# --- End of one-time setup ---

def run_experiments():
    # set experiment parameters
    max_length = 50
    label_pad = 14
    input_dim = 39
    num_classes = 29
    EPOCHS = 50 # Epochs per trial for the hyperparameter search.
    model_type = 'transformer' # 'rnn' or 'transformer'
    load_model = True
    monitor = 'val_loss'
    restore = False # Set to False to start a new training run.
    drop_out = 0.3
    size = -1

    # --- Single Experiment Run ---
    # Define the specific hyperparameters for this run
    config = {
        "model_type": "transformer_tf",
        "layer_size": 3,
        "num_heads": 1,
        "d_model": 128,
        "d_ff": 512,
        "drop_out": 0.3
    }
    learning_rate = 0.001  # This will be the peak LR for the schedule
    optimizer_name = 'adamw' # Using AdamW as specified in the previous search
    batch_size = 64 # Using a default batch size

    data = getDataTrain1(label_pad, max_length, num_classes)

    # Construct the experiment name
    exp_name = (f"L{config['layer_size']}_H{config['num_heads']}_D{config['d_model']}_F{config['d_ff']}_"
                f"Dr{int(config['drop_out']*100)}_LR{learning_rate}_{optimizer_name.upper()}_BS{batch_size}_SchedLR")

    print(f"\n{'='*20} RUNNING EXPERIMENT: {exp_name} {'='*20}")
    
    # Define a unique directory for this specific run
    run_dir = f"models/hparams/{datetime.now().strftime('%Y%m%d-%H%M%S')}_{exp_name}"
    SAVE_PATH = f"{run_dir}/" # Checkpoints will be saved in a subfolder

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    # Efficiently set up file logging for this specific trial
    log_file = os.path.join(SAVE_PATH, 'log_valid.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('[[%(asctime)s]] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    with tf.summary.create_file_writer(run_dir).as_default():
        # Log the hyperparameters for this specific run
        hparams = {
            HP_MODEL_TYPE: config.get('model_type', 'N/A'),
            HP_LEARNING_RATE: learning_rate,
            HP_DROPOUT: config.get('drop_out', 0.0),
            HP_LAYER_SIZE: config.get('layer_size', 0),
            HP_D_MODEL: config.get('d_model', 0),
            HP_NUM_HEADS: config.get('num_heads', 0),
            HP_D_FF: config.get('d_ff', 0),
            HP_OPTIMIZER: optimizer_name,
            HP_BATCH_SIZE: batch_size
        }
        hp.hparams(hparams)

        # Run training and evaluation
        train(input_dim, num_classes, learning_rate, data, batch_size, size, EPOCHS, SAVE_PATH, restore, logging, max_length, label_pad, optimizer_name=optimizer_name, **config)
        final_metrics = eval(SAVE_PATH, batch_size, logging, label_pad, max_length, num_classes)

        # Log the final metrics to associate them with the hparams
        if final_metrics:
            tf.summary.scalar(METRIC_CER, final_metrics[0], step=1) # Assumes CER is the first element
            tf.summary.scalar(METRIC_WER, final_metrics[1], step=1) # Assumes WER is the second element
    
    # Clean up the file handler for this trial to avoid logging to multiple files
    root_logger.removeHandler(file_handler)

if __name__ == "__main__":
    run_experiments()
