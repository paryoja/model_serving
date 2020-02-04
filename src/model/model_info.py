import datetime
import json
import pathlib

import numpy as np
from tensorflow import keras

from train import data_loader


def format_example(image, label):
    # image is already in range 0 ~ +1 -> change it to -1 ~ +1
    image = (image * 2) - 1
    return image, label


def simple_label(prediction, class_names):
    return class_names[np.argmax(prediction).astype('int32')]


class ModelInfo:
    def __init__(self, base_model_name, base_model_only, model_name, class_names, version, data_info, img_size,
                 create_dir=True):
        """
           :param base_model_name: Pre-trained 된 모델 이름
           :param base_model_only: base model 자체만 사용하는 경우 - Imagenet classification 자체로
           :param model_name: Serving 할 때 쓸 모델명
           :param class_names: class 명 list
           :param version: version 명
           :param data_info: Data 정보
        """
        self.base_model_name = base_model_name
        self.base_model_only = base_model_only
        self.model_name = model_name
        self.class_names = class_names
        self.version = version
        self.data_info = data_info
        self.img_size = img_size

        self.base_model = self.get_base_model(base_model_name, base_model_only, data_info)
        self.format_fn = self.get_format_fn(self.base_model_name)
        self.label_fn = self.get_label_fn()

        if create_dir:
            self.model_dir, self.graph_dir = self.make_save_dir()
            self.save()

    def save(self):
        with open(self.model_dir / 'model_info.json', 'w') as w:
            contents = {
                "base_model_name": self.base_model_name,
                "base_model_only": self.base_model_only,
                "model_name": self.model_name,
                "class_names": [name for name in self.class_names],
                "version": self.version,
                "data_info.img_size": self.data_info.img_size,
                "data_info.data_type.data_str": self.data_info.data_type.data_str,
                "img_size": self.img_size,
            }
            json.dump(contents, w)

    @classmethod
    def load(cls, filepath):
        with open(filepath) as f:
            contents = json.load(f)

        data_type = data_loader.DataType(data_str=contents["data_info.data_type.data_str"])
        data_info = data_loader.DatasetInfo(img_size=contents["data_info.img_size"],
                                            data_type=data_type)
        model_info = ModelInfo(base_model_name=contents["base_model_name"],
                               base_model_only=contents["base_model_only"],
                               model_name=contents["model_name"],
                               class_names=np.array(contents["class_names"]),
                               version=contents["version"],
                               data_info=data_info,
                               img_size=contents["img_size"],
                               create_dir=False)
        return model_info

    @staticmethod
    def get_format_fn(base_model_name):
        return format_example

    @staticmethod
    def get_label_fn():
        return simple_label

    @staticmethod
    def get_base_model(base_model_name, base_model_only, data_info):
        img_shape = (data_info.img_size, data_info.img_size, 3)
        if base_model_name == "VGG16":
            base_model = keras.applications.VGG16(input_shape=img_shape,
                                                  include_top=base_model_only,
                                                  weights='imagenet')

        elif base_model_name == "MobileNetV2":
            base_model = keras.applications.MobileNetV2(input_shape=img_shape,
                                                        include_top=base_model_only,
                                                        weights='imagenet')

        elif base_model_name == "Xception":
            base_model = keras.applications.Xception(input_shape=img_shape,
                                                     include_top=base_model_only,
                                                     weights='imagenet')

        elif base_model_name == "NASNetMobile":
            base_model = keras.applications.NASNetMobile(input_shape=img_shape,
                                                         include_top=base_model_only,
                                                         weights='imagenet')

        elif base_model_name == "InceptionResNetV2":
            base_model = keras.applications.InceptionResNetV2(input_shape=img_shape,
                                                              include_top=base_model_only,
                                                              weights='imagenet')
        else:
            raise KeyError("Unknown model name {}".format(base_model_name))
        return base_model

    def make_save_dir(self):
        model_directory = pathlib.Path('models') / self.model_name / self.version / datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S")
        graph_directory = pathlib.Path('graph') / self.model_name / self.version

        model_directory.mkdir(parents=True, exist_ok=True)
        graph_directory.mkdir(parents=True, exist_ok=True)

        return model_directory, graph_directory
