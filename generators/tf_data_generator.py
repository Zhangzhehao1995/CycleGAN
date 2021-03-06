# -*- coding: utf-8 -*-
import os.path
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from .data_augmentation import rotate, zoom, shift, no_action


def load_npy(item):
    data = np.load(item.numpy())
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    return data


def read_npy_file_label(item, classes):
    data = tf.py_function(load_npy, [item], tf.float32)
    data = tf.expand_dims(data, axis=0)
    data = data/255.0*classes
    data = tf.cast(data, tf.uint8)
    return data


def read_npy_file_image(item):
    data = tf.py_function(load_npy, [item], tf.float32)
    data = tf.expand_dims(data, axis=0)
    #data = data/255.0
    return data


def decode_jpg_and_convert(item, data_channels):
    # Convert grey image into RGB with channels=3 and uint8 form (0-255)
    img = tf.image.decode_jpeg(item, channels=data_channels)
    # img = tf.image.convert_image_dtype(img, tf.float32) # float: [0,1)
    return img


def read_jpg_file_image(item, data_channels, load_size, target_size):
    # read the JPG file, convert to [0,1] float32 and resize to load_size and crop to target_size
    img = tf.io.read_file(item)
    img = decode_jpg_and_convert(img, data_channels)
    img = tf.image.resize(img, load_size)
    if load_size != target_size:
        img = tf.image.random_crop(img, list(target_size) + [tf.shape(img)[-1]])
    # convert to (-1， 1)
    img = tf.clip_by_value(img, 0, 255) / 255.0
    img = img * 2 - 1
    return img


def make_tf_dataset(file_path, data_dir, data_subdir, data_suffix):
    image_files = []
    fp = open(file_path)
    lines = fp.readlines()
    fp.close()
    nb_sample = len(lines)

    for line in lines:
        line = line.strip('\n')
        image_files.append(os.path.join(data_dir, data_subdir, line + data_suffix))

    image_files_ds = tf.data.Dataset.from_tensor_slices(image_files)
    return image_files_ds


def file2image_dataset(image_files_ds, data_channels, load_size, target_size, data_suffix, ro_range, zoom_range, dx, dy, dz, aug):
    if data_suffix == '.npy':
        image_ds = image_files_ds.map(lambda item: read_npy_file_image(item))
    elif data_suffix == '.jpg':
        image_ds = image_files_ds.map(lambda item: read_jpg_file_image(item, data_channels, load_size, target_size))
    else:
        print('ERROR: unknown data_suffix: ' + data_suffix)
        exit()

    # data augmentation (rotate, zoom, shift)
    if aug:
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


def tfDataGenerator(file_pathA, file_pathB, data_dir, data_subdirs, data_suffix, data_channels, load_size, target_size, batch_size, shuffle, ro_range, zoom_range, dx, dy, dz, aug):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    fileA_ds = make_tf_dataset(file_pathA, data_dir, data_subdirs[0], data_suffix)
    fileB_ds = make_tf_dataset(file_pathB, data_dir, data_subdirs[1], data_suffix)

    samplesA_ds = len(["" for line in open(file_pathA, "r")])
    samplesB_ds = len(["" for line in open(file_pathB, "r")])

    fileA_ds = fileA_ds.shuffle(buffer_size=samplesA_ds)
    fileB_ds = fileB_ds.shuffle(buffer_size=samplesB_ds)

    imageA_ds = file2image_dataset(fileA_ds, data_channels, load_size, target_size, data_suffix, ro_range, zoom_range, dx, dy, dz, aug)
    imageB_ds = file2image_dataset(fileB_ds, data_channels, load_size, target_size, data_suffix, ro_range, zoom_range, dx, dy, dz, aug)

    if samplesA_ds > samplesB_ds:
        imageB_ds = imageB_ds.repeat(int(np.ceil(samplesA_ds / samplesB_ds)))
    else:
        imageA_ds = imageA_ds.repeat(int(np.ceil(samplesB_ds / samplesA_ds)))

    images_ds = tf.data.Dataset.zip((imageA_ds, imageB_ds))

    images_ds = images_ds.batch(batch_size)
    images_ds = images_ds.repeat()
    images_ds = images_ds.prefetch(buffer_size=AUTOTUNE)

    return images_ds
