import datetime
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.models import model_from_json

from train import data_loader


class TrainInfo:
    def __init__(self, learning_rate=0.0001, momentum=0.9, update_base=False, shuffle_buffer_size=1000, batch_size=92):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.update_base = update_base
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size


def format_example(image, label):
    # image is already in range 0 ~ +1 -> change it to -1 ~ +1
    image = (image * 2) - 1
    return image, label


def simple_label(prediction, class_names):
    return class_names[np.argmax(prediction).astype('int32')]


def print_min_max(image):
    print("min", tf.math.reduce_min(image))
    print("max", tf.math.reduce_max(image))


class ModelInfo:
    def __init__(self, base_model_name, base_model_only, model_name, class_names, version, data_info, create_dir=True):
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
            }
            json.dump(contents, w)

    @classmethod
    def load(cls, filepath):
        with open(filepath) as f:
            contents = json.load(f)

        data_info = data_loader.DatasetInfo(img_size=contents["data_info.img_size"])
        model_info = ModelInfo(base_model_name=contents["base_model_name"],
                               base_model_only=contents["base_model_only"],
                               model_name=contents["model_name"],
                               class_names=np.array(contents["class_names"]),
                               version=contents["version"],
                               data_info=data_info,
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


class DeepModel:
    def __init__(self, model_info, *args):

        self.model_info = model_info

        if not self.model_info.base_model_only:
            self.model = self.make_model(*args)
        else:
            self.model = self.model_info.base_model

    def make_model(self, *args):
        raise NotImplementedError()

    def make_callbacks(self, train_info):
        cb_lrate = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20, verbose=1,
                                                     mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
        cb_tb_hist = keras.callbacks.TensorBoard(log_dir=str(self.model_info.graph_dir), histogram_freq=0,
                                                 write_graph=True, write_images=False, profile_batch=None)
        cb_checkpoint = keras.callbacks.ModelCheckpoint(
            str(self.model_info.model_dir / './model.{epoch:03d}-{val_loss:.2f}.hdf5'), verbose=1,
            monitor='val_loss',
            save_best_only=True, mode='auto')
        return [cb_lrate, cb_tb_hist, cb_checkpoint]

    def train_model(self, dataset, train_info: TrainInfo):
        raw_train, raw_val = dataset.get_raw_data()

        for image in raw_train.take(1):
            print("Min, Max for raw train")
            print_min_max(image[0])

        for image in raw_val.take(1):
            print("Min, Max for raw val")
            print_min_max(image[0])

        train = raw_train.map(self.model_info.format_fn)
        val = raw_val.map(self.model_info.format_fn)

        self.model_info.base_model.trainable = train_info.update_base

        # 일단 여기서 cache 해본다 -> 메모리 부족시 앞에서 해본다
        train_batches = train.cache().shuffle(train_info.shuffle_buffer_size).batch(train_info.batch_size)
        val_batches = val.cache().batch(train_info.batch_size)

        for image in train_batches.take(1):
            print("Min, Max for train batches")
            print_min_max(image[0])

        for image in val_batches.take(1):
            print("Min, Max for val batches")
            print_min_max(image[0])

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(lr=train_info.learning_rate, momentum=train_info.momentum),
            loss="categorical_crossentropy",
            metrics=['accuracy'])
        self.save()
        self.model.summary()

        initial_epochs = 200

        loss0, accuracy0 = self.model.evaluate(val_batches)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        callbacks = self.make_callbacks(train_info)

        try:
            history = self.model.fit(train_batches,
                                     epochs=initial_epochs,
                                     validation_data=val_batches,
                                     callbacks=callbacks,
                                     verbose=1)

            write_history(history)
        except KeyboardInterrupt as e:
            print(e)

    def load_file(self, model_file, json_file):
        with open(json_file) as f:
            json_model = f.read()
        self.model = model_from_json(json_model)
        self.model.load_weights("{}".format(model_file))

    def validate(self, dataset):
        raw_train, raw_val = dataset.get_raw_data()

        train = raw_train.map(self.model_info.format_fn)
        val = raw_val.map(self.model_info.format_fn)

        train_loss, train_accuracy = self.model.evaluate(train)
        val_loss, val_accuracy = self.model.evaluate(val)

        print("train loss: {:.2f}".format(train_loss))
        print("train accuracy: {:.2f}".format(train_accuracy))
        print("val loss: {:.2f}".format(val_loss))
        print("val accuracy: {:.2f}".format(val_accuracy))

    def predict(self, np_image):
        formatted_image = self.model_info.format_fn(np_image, "?")

        predictions = self.model.predict(tf.expand_dims(formatted_image[0], axis=0))
        if self.model_info.base_model_only:
            label = decode_predictions(predictions)
        else:
            label = self.model_info.label_fn(predictions, self.model_info.class_names)
        version = self.model_info.version

        prob_list = []
        for prob in predictions[0]:
            prob_list.append(prob.astype(float))
        return prob_list, label, version

    def save(self):
        with open(self.model_info.model_dir / "model.json", "w") as json_file:
            model_json = self.model.to_json()
            json_file.write(model_json)
        # self.model.save_weights('final_model', save_format='tf')


class LoadedModel(DeepModel):
    def make_model(self, *args):
        # 첫번째 args 자체가 모델
        return args[0]


def write_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    plt.savefig('accuracy_{}.png'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
