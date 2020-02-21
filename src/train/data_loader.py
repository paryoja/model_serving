import functools
import os
import pathlib
import time

import numpy as np
import tensorflow as tf

from train import data_downloader
from utils.model_logger import logger


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


def debug_path(file_path, class_names):
    label = get_label(file_path, class_names)
    img = tf.io.read_file(file_path)

    return label, file_path


def process_path(file_path, class_names, img_size, keep_path):
    label = get_label(file_path, class_names)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_size)

    if keep_path:
        return (img, file_path), label
    else:
        return img, label


class DataType:
    PokemonYesNo = "pokemon_yes_no"
    People = "People"

    def __init__(self, data_str):
        self.data_str = data_str

    def download(self, session):
        file_type = self.load_file_type()
        if self.data_str == DataType.PokemonYesNo:
            data_downloader.download_pokemon(session, file_type, label="yes")
            data_downloader.download_pokemon(session, file_type, label="no")
            data_downloader.download_pokemon(session, file_type, label="more")
            data_downloader.download_pokemon(session, file_type, label="little")
            data_downloader.validate_image(self.data_str)
        elif self.data_str == DataType.People:
            data_downloader.download_people(session, file_type, label="True")
            data_downloader.download_people(session, file_type, label="False")

    def load_file_type(self):
        file_type = {}
        if self.data_str == DataType.People:
            glob = pathlib.Path('data/people').glob("**/*.jpg")

            for file in glob:
                if file.name in file_type:
                    logger.warning("Duplicated filename {} deleting it".format(file.name))
                    logger.warning("Unlink {} and {}".format(file, file_type[file.name][1]))
                    file.unlink()
                    file_type[file.name][1].unlink()
                    del file_type[file.name]
                    continue
                file_type[file.name] = (file.parts[-2], file)
        return file_type


def confusing_data(model, data_info):
    start_time = time.time()
    dataset = Dataset(data_info, keep_path=True)
    train_ds = dataset.train_ds.prefetch(100000)
    need_to_train_path = []
    threshold = 0.8
    for images_path, label in train_ds.batch(500):
        predict = model.predict_batch(images_path[0])

        for lab, p, path in zip(label, predict, images_path[1]):
            prob = tf.squeeze(tf.boolean_mask(p, lab))

            if prob < threshold:
                need_to_train_path.append(path)

    logger.info("Finished classifying train data : Total {}".format(len(need_to_train_path)))
    with open("more_train.txt", "w") as w:
        for path in need_to_train_path:
            w.write(path.numpy().decode('utf-8'))
            w.write("\n")
    logger.info("elapsed time", time.time() - start_time)
    logger.info("train count", len(need_to_train_path))

    return need_to_train_path


class DatasetInfo:
    def __init__(self, data_type, img_size):
        self.data_type = data_type
        self.img_size = img_size

    def __str__(self):
        return "{} {}".format(self.data_type, self.img_size)


class Dataset:
    def __init__(self, dataset_info, keep_path=False, from_untrained_file=False):

        if from_untrained_file:
            need_to_train_path = []
            with open("more_train.txt") as f:
                for line in f:
                    need_to_train_path.append(line.strip())
            list_ds = tf.data.Dataset.from_tensor_slices(need_to_train_path)

            self.data_count = len(need_to_train_path)
        else:
            data_dir = pathlib.Path('data/{}/train/'.format(dataset_info.data_type.data_str))

            list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))
            self.data_count = len(list(data_dir.glob('**/*')))

        val_dir = pathlib.Path('data/{}/validate/'.format(dataset_info.data_type.data_str))
        val_list_ds = tf.data.Dataset.list_files(str(val_dir / '*/*'))

        self.class_names = np.array([item.name for item in val_dir.glob('*') if item.name != "LICENSE.txt"])
        logger.debug("Class names {}".format(self.class_names))
        self.validate_count = len(list(val_dir.glob('**/*')))

        logger.info("Data count {}, Validate count {}".format(self.data_count, self.validate_count))
        debug = False
        if not debug:
            process_path_partial = functools.partial(process_path, class_names=self.class_names,
                                                     img_size=dataset_info.img_size, keep_path=keep_path)

        else:
            process_path_partial = functools.partial(debug_path, class_names=self.class_names)

        self.train_ds = list_ds.map(process_path_partial, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.val_ds = val_list_ds.map(process_path_partial, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # import matplotlib.pyplot as plt
        # for img in self.train_ds:
        #     plt.imshow(img[0])
        #     break

    def get_raw_data(self):
        return self.train_ds, self.val_ds
