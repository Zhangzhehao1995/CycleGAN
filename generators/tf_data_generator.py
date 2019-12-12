# -*- coding: utf-8 -*-
import os.path
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from .data_augmentation import rotate,zoom,shift,no_action
#tf.enable_eager_execution()


def load_npy(item):
    data=np.load(item.numpy())
    data=tf.convert_to_tensor(data, dtype=tf.float32)
    return data

def read_npy_file_image(item):
    data = tf.py_function(load_npy, [item], tf.float32)
    data = tf.expand_dims(data, axis=0)
    #data = data/255.0
    return data

def decode_jpg_and_convert(item):
    img = tf.image.decode_jpeg(item)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def read_jpg_file_image(item, target_size):
    # read the JPG file, convert to [0,1] float32 and resize to target_size
    img = tf.io.read_file(item)
    img = decode_jpg_and_convert(img)
    img = tf.image.resize(img, target_size)
    return img

def read_npy_file_label(item,classes):
    data = tf.py_function(load_npy,[item],tf.float32)
    data = tf.expand_dims(data,axis=0)
    data = data/255.0*classes
    data = tf.cast(data,tf.uint8)
    return data

def make_tf_dataset(file_path, data_dir, data_subdir, target_size, data_suffix, ro_range, zoom_range, dx, dy, dz, aug):
    image_files = []
    fp = open(file_path)
    lines = fp.readlines()
    fp.close()
    nb_sample = len(lines)

    for line in lines:
        line = line.strip('\n')
        image_files.append(os.path.join(data_dir, data_subdir, line + data_suffix))

    image_files_ds = tf.data.Dataset.from_tensor_slices(image_files)
    if data_suffix == '.npy':
        image_ds = image_files_ds.map(lambda item: read_npy_file_image(item))
    elif data_suffix == '.jpg':
        image_ds = image_files_ds.map(lambda item: read_jpg_file_image(item, target_size))
    else:
        print('ERROR: unknown data_suffix: ' + data_suffix)
        exit()

    # data augmentation (rotate, zoom, shift)
    if aug == True:
        image_ds = image_ds.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
                                lambda: rotate(x, ro_range=ro_range),
                                lambda: no_action(x)))
        image_ds = image_ds.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
                                lambda: zoom(x, zoom_range=zoom_range, target_size=target_size),
                                lambda: no_action(x)))
        image_ds = image_ds.map(lambda x: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
                                lambda: shift(x, dx_range=dx, dy_range=dy, dz_range=dz),
                                lambda: no_action(x)))
    return image_ds

def tfDataGenerator(file_pathA, file_pathB, data_dir, data_suffix, data_subdirs, target_size, batch_size, shuffle, ro_range, zoom_range, dx, dy, dz, aug):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    imageA_ds = make_tf_dataset(file_pathA, data_dir, data_subdirs[0], target_size, data_suffix, ro_range, zoom_range, dx, dy, dz, aug)
    imageB_ds = make_tf_dataset(file_pathB, data_dir, data_subdirs[1], target_size, data_suffix, ro_range, zoom_range, dx, dy, dz, aug)
    imageA_ds = imageA_ds.shuffle(buffer_size=batch_size)
    imageB_ds = imageB_ds.shuffle(buffer_size=batch_size)

    images_ds = tf.data.Dataset.zip((imageA_ds,imageB_ds))

    images_ds=images_ds.batch(batch_size)
    images_ds=images_ds.repeat()
    images_ds=images_ds.prefetch(buffer_size=AUTOTUNE)
    #iterator=ds.make_one_shot_iterator()

    return images_ds
