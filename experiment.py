#!/usr/bin/env python

""" training script for cycleGAN
"""

from os.path import join
from os import path, mkdir, getcwd, chdir, makedirs
from getpass import getuser
import sys
import numpy as np
#import matplotlib.pyplot as plt
from argparse import ArgumentParser
#matplotlib.use('Agg')  # Faster plot

# Import tools
from configuration.configuration import Configuration
from utils.logger import Logger

# network
from callbacks.callbacks_factory import Callbacks_Factory

from generators.dataset_generators import Dataset_Generators
from models.cyclegan import CycleGAN
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import tensorflow as tf
#tf.enable_eager_execution()

""" temporary fix, only needed for non-cluster tf if receiving the following error:
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Experiment(object):
    """Class to load configuration and run experiment."""

    def __init__(self):
        self.cf = None

    def run(self, configuration):
        # Load configuration
        self.cf = configuration.load()
        self.cf.data_format=K.image_data_format()
        if self.cf.train_model:
            # Enable log file
            sys.stdout = Logger(self.cf.log_file)
        print (' ---> Starting experiment: ' + self.cf.experiment_name + ' <---')

        # Create the data generators
        train_gen, test_gen = Dataset_Generators().make(self.cf)

        # Build model
        print ('\n > Building model...')
        model = CycleGAN(self.cf)
        model.make()

        # Create the callbacks
        print ('\n > Creating callbacks...')
        cb = Callbacks_Factory().make(self.cf)

        try:
            if self.cf.train_model:
                # Train the model
                model.train(train_gen, cb)
                # losses saved in train function to JSON
                #loss_value=hist.history['loss']
                #val_loss_value=hist.history['val_loss']
                #np.save(join(cf.savepath,'loss_' + str(cf.experiment_name)+ '.npy'),loss_value)
                #np.save(join(cf.savepath,'val_loss_' + str(cf.experiment_name)+ '.npy'),val_loss_value)

            if self.cf.test_model:
                # Compute test metrics
                model.test(test_gen)

            if self.cf.pred_model:
                # Compute test metrics
                model.predict(test_gen, tag='pred')

        except KeyboardInterrupt:
            # In case of early stopping, transfer the local files
            do_copy = input('\033[93m KeyboardInterrupt \nDo you want to transfer files to {} ? ([y]/n) \033[0m'
                                .format(self.cf.final_savepath))
            if do_copy in ['', 'y']:
                # Copy result to shared directory
                configuration.copy_to_shared()
            raise

        # Finish
        print (' ---> Finished experiment: ' + self.cf.experiment_name + ' <---')

# Entry point of the script
if __name__ == "__main__":

    parser = ArgumentParser(description='Model training')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-d', '--data_path', type=str, help='Path to data', required=True)
    parser.add_argument('-e', '--experiment_path', type=str, help='Path to experiments (output)', required=True)
    args = parser.parse_args()

    # paths
    dataset_path = args.data_path
    experiments_path = args.experiment_path
    config_file = args.config_file
    usr_path = join(path.expanduser("~"), getuser())

    #tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    #tf_session = tf.Session() #config=tf_config)
    #K.set_session(tf_session)

    # Load configuration files
    configuration = Configuration(config_file,
                                  dataset_path,
                                  experiments_path,
                                  usr_path)

    # Train /test/predict with the network, depending on the configuration
    experiment = Experiment()
    experiment.run(configuration)

    # Copy result to shared directory
    configuration.copy_to_shared()
