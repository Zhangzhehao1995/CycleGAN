#from .seg_data_generator import SegDataGenerator
from .tf_data_generator import tfDataGenerator

# Load datasets

class Dataset_Generators():
    def __init__(self):
        pass

    def make(self, cf):
        # mean = cf.dataset.rgb_mean
        # std = cf.dataset.rgb_std
        # cf.dataset.cb_weights = None
        ds_train=None
        ds_test=None
        if cf.train_model:
            # Load training set
            print ('\n > Reading training set...')
            # Create the data generator with its data augmentation
            ds_train=tfDataGenerator(file_pathA=cf.trainA_file_path_full,
                                      file_pathB=cf.trainB_file_path_full,
                                      data_dir=cf.image_path_full,
                                      data_channels = cf.channel_size,
                                      data_suffix=cf.data_suffix,
                                      data_subdirs=cf.image_path_subdirs,
                                      target_size=cf.target_size_train,
                                      shuffle_buffer=cf.shuffle_buffer,
                                      batch_size=cf.batch_size_train,
                                      shuffle=cf.shuffle_train,
                                      ro_range=cf.da_rotation_range,
                                      zoom_range=cf.da_zoom_range,
                                      dx=cf.da_height_shift_range,
                                      dy=cf.da_width_shift_range,
                                      dz=cf.da_depth_shift_range,
                                      aug=True
                                      )


        if cf.test_model or cf.pred_model:
            # Load testing set
            print ('\n > Reading testing set...')
            ds_test=tfDataGenerator(file_pathA=cf.testA_file_path_full,
                                      file_pathB=cf.testB_file_path_full, 
                                      data_dir=cf.image_path_full,
                                      data_channels=cf.channel_size,
                                      data_suffix=cf.data_suffix,
                                      data_subdirs=cf.image_path_subdirs,
                                      target_size=cf.target_size_test,
                                      shuffle_buffer=cf.shuffle_buffer,
                                      batch_size=cf.batch_size_test,
                                      shuffle=cf.shuffle_train,
                                      ro_range=cf.da_rotation_range,
                                      zoom_range=cf.da_zoom_range,
                                      dx=cf.da_height_shift_range,
                                      dy=cf.da_width_shift_range,
                                      dz=cf.da_depth_shift_range,
                                      aug=False)
        else:
            ds_test = None

        return (ds_train, ds_test)
