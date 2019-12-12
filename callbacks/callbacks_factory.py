import math
import os

from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, CSVLogger,
                             LearningRateScheduler, TensorBoard)

from .callbacks import (LearningRateSchedulerBatch, Scheduler)


# Create callbacks
class Callbacks_Factory():
    def __init__(self):
        pass

    def make(self, cf):
        cb = []

        # Early stopping
        if cf.earlyStopping_enabled:
            print('   Early stopping')
            cb += [EarlyStopping(monitor=cf.earlyStopping_monitor,
                                 mode=cf.earlyStopping_mode,
                                 patience=cf.earlyStopping_patience,
                                 verbose=cf.earlyStopping_verbose)]

        # Define model saving callbacks
        if cf.checkpoint_enabled:
            print('   Model Checkpoint')
            cb += [ModelCheckpoint(filepath=os.path.join(cf.savepath, "weights-{epoch:02d}.hdf5"),
                                   verbose=cf.checkpoint_verbose,
                                   monitor=cf.checkpoint_monitor,
                                   mode=cf.checkpoint_mode,
                                   save_best_only=cf.checkpoint_save_best_only,
                                   save_weights_only=cf.checkpoint_save_weights_only,
                                   period=20)]


        # Learning rate scheduler
        if cf.LRScheduler_enabled:
            print('   Learning rate scheduler by batch')
            scheduler = Scheduler(cf.LRScheduler_type, cf.LRScheduler_power, cf.n_epochs, cf.learning_rate_base)

            if cf.LRScheduler_batch_epoch == 'batch':
                cb += [LearningRateSchedulerBatch(scheduler.scheduler_function)]
            elif cf.LRScheduler_batch_epoch == 'epoch':
                cb += [LearningRateScheduler(scheduler.scheduler_function)]
            else:
                raise ValueError('Unknown scheduler mode: ' + LRScheduler_batch_epoch)


        # Output the list of callbacks
        return cb
