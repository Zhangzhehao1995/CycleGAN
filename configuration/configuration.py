from __future__ import print_function

import imp
import time
import os
from distutils.dir_util import copy_tree
import shutil


class Configuration():
    def __init__(self, config_file,
                       dataset_path,
                       experiments_path,
                       usr_path):

        self.config_file = config_file
        self.dataset_path = dataset_path
        self.experiments_path = experiments_path
        self.usr_path = usr_path

    def load(self):
        # Load configuration file
        print(self.config_file)
        cf = imp.load_source('config', self.config_file)

        # Save extra parameter
        cf.config_file = self.config_file

        # Create output folders
        cf.savepath = os.path.join(self.experiments_path, cf.dataset_name, cf.experiment_name)
        cf.final_savepath = os.path.join(self.experiments_path, cf.dataset_name,
                                         cf.experiment_name)
        cf.log_file = os.path.join(cf.savepath, "logfile.log")
        if not os.path.exists(cf.savepath):
            os.makedirs(cf.savepath)
        cf.usr_path = self.usr_path

        # Copy config file
        shutil.copyfile(self.config_file, os.path.join(cf.savepath, "config.py"))


        # If in Debug mode use few images
        if cf.debug and cf.debug_images_train > 0:
            cf.n_images_train = cf.debug_images_train
        else:
            cf.n_images_train = None

        if cf.debug and cf.debug_images_test > 0:
            cf.n_images_test = cf.debug_images_test
        else:
            cf.n_images_test = None

        if cf.debug and cf.debug_n_epochs > 0:
            cf.n_epochs = cf.debug_n_epochs

        # learning rate
        #cf.learning_rate_base =cf.lr_base* (float(cf.batch_size_train) / 16)

        # Dataset
        # Define target sizes
        if cf.crop_size_train is not None: cf.target_size_train = cf.crop_size_train
        if cf.crop_size_test is not None: cf.target_size_test = cf.crop_size_test
        if cf.target_size_train:
            cf.input_shape = cf.target_size_train + (cf.channel_size,)
        else:
            cf.input_shape = (None, None, cf.channel_size)

        # load_size, zzh
        if cf.load_size_train is None: cf.load_size_train = cf.target_size_train
        if cf.load_size_test is None: cf.load_size_test = cf.target_size_test

        # paths
        cf.image_path_full = os.path.join(self.dataset_path, cf.image_path)

        cf.trainA_file_path_full = os.path.join(self.dataset_path, cf.trainA_file_path)
        cf.testA_file_path_full = os.path.join(self.dataset_path, cf.testA_file_path)
        cf.trainB_file_path_full = os.path.join(self.dataset_path, cf.trainB_file_path)
        cf.testB_file_path_full = os.path.join(self.dataset_path, cf.testB_file_path)

        # batch sizes
        cf.batch_shape_train = (cf.batch_size_train,) + cf.input_shape

        # test batch_size always be 1
        if cf.batch_size_test != 1:
            cf.batch_size_test = 1
        cf.batch_shape_test = (cf.batch_size_test,) + cf.input_shape

        # Get weights file name
        # path, _ = os.path.split(cf.weights_file)
        # if path == '':
        #     cf.weights_file = os.path.join(cf.savepath, cf.weights_file)

        # cf.checkpoint_path = os.path.join(cf.savepath, cf.pretrained_file)
        # zzh
        if cf.load_epoch_for_test is None:
            cf.load_epoch_for_test = cf.n_epochs

        self.configuration = cf
        return cf

    # Copy result to shared directory
    def copy_to_shared(self):
        if self.configuration.savepath != self.configuration.final_savepath:
            print('\n > Copying model and other training files to {}'.format(self.configuration.final_savepath))
            start = time.time()
            copy_tree(self.configuration.savepath, self.configuration.final_savepath)
            open(os.path.join(self.configuration.final_savepath, 'lock'), 'w').close()
            print ('   Copy time: ' + str(time.time()-start))
