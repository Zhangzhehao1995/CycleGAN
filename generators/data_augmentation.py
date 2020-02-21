# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

#tf.enable_eager_execution()

def rotate(x, y=None, ro_range=0.0):

    pi = tf.constant(np.pi)

    ro_range = ro_range/180*pi

    angle_a = tf.random.uniform(shape=[], minval=-ro_range, maxval=ro_range, dtype=tf.float32)
    tfa.image.transform_ops.rotate(x, angle_a)

    if y is not None:
        angle_b = tf.random.uniform(shape=[], minval=-ro_range, maxval=ro_range, dtype=tf.float32)
        angle_c = tf.random.uniform(shape=[], minval=-ro_range, maxval=ro_range, dtype=tf.float32)
        y = tf.transpose(y,[3,0,1,2])
        y = tfa.image.transform_ops.rotate(y,angle_a)
        y = tf.transpose(y,[0,1,3,2])
        y = tfa.image.transform_ops.rotate(y,angle_b)
        y = tf.transpose(y,[0,3,2,1])
        y = tfa.image.transform_ops.rotate(y,angle_c)
        y = tf.transpose(y,[3,1,2,0])

        return x, y
    else:
        return x

def zoom(x, y=None, zoom_range=(1.0, 1.0), target_size=None):
    Minval,Maxval=zoom_range
    print ('minval=',Minval)
    rate_a=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    rate_b=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)
    rate_c=tf.random.uniform(shape=[],minval=Minval,maxval=Maxval,dtype=tf.float32)

    target_height, target_width = target_size

    x = tf.image.resize(x, tf.cast((target_height*rate_a, target_width*rate_b), dtype=tf.int32))
    x = tf.image.resize_with_crop_or_pad(x, target_height, target_width)

    if y is not None:
        y=tf.transpose(y,[3,0,1,2])
        y=tf.image.resize_nearest_neighbor(y,tf.cast((target_height*rate_a,target_width*rate_a),dtype=tf.int32))
        y=tf.image.resize_image_with_crop_or_pad(y,target_height,target_width)
        y=tf.transpose(y,[0,1,3,2])
        y=tf.image.resize_nearest_neighbor(y,tf.cast((target_height*rate_b,target_depth*rate_b),dtype=tf.int32))
        y=tf.image.resize_image_with_crop_or_pad(y,target_height,target_depth)
        y=tf.transpose(y,[0,3,2,1])
        y=tf.image.resize_nearest_neighbor(y,tf.cast((target_width*rate_c,target_depth*rate_c),dtype=tf.int32))
        y=tf.image.resize_image_with_crop_or_pad(y,target_width,target_depth)
        y=tf.transpose(y,[3,1,2,0])
        y=tf.cast(y,tf.uint8)

        return x, y
    else:
        return x

def shift(x, y=None, dx_range=0.0, dy_range=0.0, dz_range=0.0):

    dx=tf.random.uniform(shape=[], minval=-dx_range, maxval=dx_range, dtype=tf.float32)
    dy=tf.random.uniform(shape=[], minval=-dy_range, maxval=dy_range, dtype=tf.float32)
    # dz=tf.random.uniform(shape=[], minval=-dz_range, maxval=dz_range, dtype=tf.float32)

    x=tfa.image.translate(x, [dx,dy], interpolation="BILINEAR")

    if y is not None:
        # y=tf.transpose(y,[3,0,1,2])
        y=tfa.image.translate(y, [dx,dy], interpolation="NEAREST")
        # y=tf.transpose(y,[0,1,3,2])
        # y=tf.contrib.image.translate(y,[dx,dz])
        # y=tf.transpose(y,[0,3,2,1])
        # y=tf.contrib.image.translate(y,[dy,dz])
        # y=tf.transpose(y,[3,1,2,0])

        return x, y
    else:
        return x

def no_action(x, y=None):
    if y is not None:
        return x,y
    else:
        return x
