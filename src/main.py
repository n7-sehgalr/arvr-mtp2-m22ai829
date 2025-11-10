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

    logging.basicConfig(
        filename=log_file,
        level=logging_level,
        format='[[%(asctime)s]] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    return logging

def main(size,log,SAVE_PATH,model_size=400,layer_size = 2,drop_out= 0.2):
    data = getDataTrain1(label_pad,max_length,num_classes)
    train(input_dim, num_classes, learning_rate,data,
          batch_size, size, EPOCHS, SAVE_PATH, 
          restore,log,max_length,label_pad,
          model_size=model_size,layer_size= layer_size,drop_out = drop_out)




if __name__ == "__main__":
    # set experiment parameters
    max_length = 50
    label_pad = 14
    input_dim = 39
    num_classes = 29
    learning_rate = 0.00001 # Changed from 0.0001. It's often good practice to lower the learning rate when resuming training
    batch_size = 64
    EPOCHS = 1 # Set to 1 to train for a single epoch after resuming from 200th epoch
    load_model = True

    # SAVE_PATH = datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
    #  To resume, point SAVE_PATH to the existing model directory
    SAVE_PATH = '20251110-012452/' 
    monitor = 'val_loss'
    restore = True # Set to false when training from scratch
    drop_out = 0.2

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    log = init_logging(SAVE_PATH)

    size = -1

    main(size, log, SAVE_PATH)
    eval(SAVE_PATH, batch_size, log, label_pad, max_length, num_classes)