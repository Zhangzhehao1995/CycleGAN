# Imports
from tensorflow.keras.callbacks import (Callback, LearningRateScheduler)

import numpy as np
import time
import math

class Scheduler():
    """ Learning rate scheduler function
    # Arguments
        scheduler_mode: ['linear' | 'step' | 'square' | 'sqrt']
        lr: initial learning rate
        M: number of learning iterations
        decay: decay coefficient
        S: step iteration
        from: https://arxiv.org/pdf/1606.02228.pdf
        poly from: https://arxiv.org/pdf/1606.00915.pdf
    """
    def __init__(self, scheduler_mode='power_decay', lr_power=0.9, epochs=250, lr_base=0.005):
        # Save parameters
        self.scheduler_mode = scheduler_mode
        self.lr_power = float(lr_power)
        self.lr_base = float(lr_base)
        self.epochs = float(epochs)

        # Get function
        if self.scheduler_mode == 'power_decay':
            self.scheduler_function = self.power_decay_scheduler
        elif self.scheduler_mode == 'exp_decay':
            self.scheduler_function = self.exp_decay_scheduler
        elif self.scheduler_mode == 'adam':
            self.scheduler_function = self.adam_scheduler
        elif self.scheduler_mode == 'progressive_drops':
            self.scheduler_function = self.progressive_drops_scheduler
        else:
            raise ValueError('Unknown scheduler: ' + self.scheduler_mode)

    def power_decay_scheduler(self, epoch):
        return self.lr_base * ((1 - float(epoch)/self.epochs) ** self.lr_power)

    def exp_decay_scheduler(self, epoch):
        return (self.lr_base ** self.lr_power) ** float(epoch+1)

    def adam_scheduler(self, epoch):
        return 0.001

    def progressive_drops_scheduler(self, epoch):
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * self.epochs:
            lr = 0.0001
        elif epoch > 0.75 * self.epochs:
            lr = 0.001
        elif epoch > 0.5 * self.epochs:
            lr = 0.01
        else:
            lr = 0.1
        return lr


class LearningRateSchedulerBatch(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, epochndexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateSchedulerBatch, self).__init__()
        self.schedule = schedule
        self.iter = 0

    def on_batch_begin(self, batch, logs=None):
        self.iter += 1
        self.change_lr(self.iter)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.schedule(self.iter)
        print('   New lr: ' + str(lr))

    def change_lr(self, epochteration):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(iteration)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
