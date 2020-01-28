import functools
import os
import pathlib

import numpy as np
import tensorflow as tf


def get_label(file_path, class_names):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == class_names


def decode_img(img, img_size):
    # convert the compressed string to a 3D uint8 tensor

    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.decode_image(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [img_size, img_size])


def process_path(file_path, class_names, img_size):
    label = get_label(file_path, class_names)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)
    return img, label


class DatasetInfo:
    def __init__(self, img_size):
        self.img_size = img_size


class Dataset:
    def __init__(self, dataset_info):
        data_dir = pathlib.Path('data/train')
        val_dir = pathlib.Path('data/validate/')
        self.class_names = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])

        list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))
        val_list_ds = tf.data.Dataset.list_files(str(val_dir / '*/*'))

        self.data_count = len(list(data_dir.glob('**/*')))
        self.validate_count = len(list(val_dir.glob('**/*')))

        print("Data count ", self.data_count, ", Validate count ", self.validate_count)

        process_path_partial = functools.partial(process_path, class_names=self.class_names,
                                                 img_size=dataset_info.img_size)

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.train_ds = list_ds.map(process_path_partial, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_ds = val_list_ds.map(process_path_partial, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def get_raw_data(self):
        return self.train_ds, self.val_ds
