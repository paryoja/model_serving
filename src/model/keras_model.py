import datetime
import typing

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.models import model_from_json


class DeepModel:
    def __init__(self, base_model, format_fn, base_model_only, model_name, label_fn, class_names):
        self.base_model = base_model
        self.version = "1.0"
        self.model_name = model_name
        self.base_model_only = base_model_only
        self.label_fn = label_fn
        self.class_names = class_names

        if not base_model_only:
            # TODO: 여기서 모델 추가 구현 하지 말고 다른데서?
            additional_conv_layer = keras.layers.Conv2D(32, 3, padding="same", activation='relu')
            dropout_layer_1 = keras.layers.Dropout(0.5)
            global_average_layer = keras.layers.GlobalAveragePooling2D()
            dense_layer = keras.layers.Dense(512)
            dropout_layer_2 = keras.layers.Dropout(0.5)
            prediction_layer = keras.layers.Dense(len(self.class_names))

            self.model = keras.Sequential([
                base_model,
                additional_conv_layer,
                dropout_layer_1,
                global_average_layer,
                dense_layer,
                dropout_layer_2,
                prediction_layer
            ])
        else:
            self.model = self.base_model

        self.shuffle_buffer_size = 1000
        self.batch_size = 92
        self.format_fn = format_fn

        # callbacks
        self.cb_lrate = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=20, verbose=1,
                                                          mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
        self.cb_tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True,
                                                      write_images=False, profile_batch=None)
        self.cb_checkpoint = keras.callbacks.ModelCheckpoint('./model.h5', verbose=1, monitor='val_loss',
                                                             save_best_only=True, mode='auto')

        self.callbacks = [self.cb_lrate, self.cb_tb_hist, self.cb_checkpoint]

    def train(self, dataset, update_base: bool):
        raw_train, raw_val = dataset.get_raw_data()

        for image in raw_train.take(1):
            print("Min, Max for raw train")
            print_min_max(image[0])

        for image in raw_val.take(1):
            print("Min, Max for raw val")
            print_min_max(image[0])

        train = raw_train.map(self.format_fn)
        val = raw_val.map(self.format_fn)

        self.base_model.trainable = update_base

        # 일단 여기서 cache 해본다 -> 메모리 부족시 앞에서 해본다
        train_batches = train.cache().shuffle(self.shuffle_buffer_size).batch(self.batch_size)
        val_batches = val.cache().batch(self.batch_size)

        for image in train_batches.take(1):
            print("Min, Max for train batches")
            print_min_max(image[0])

        for image in val_batches.take(1):
            print("Min, Max for val batches")
            print_min_max(image[0])

        base_learning_rate = 5e-6
        self.model.compile(optimizer=keras.optimizers.RMSprop(lr=base_learning_rate, momentum=0.9),
                           loss="categorical_crossentropy",
                           metrics=['accuracy'])
        self.save()
        self.model.summary()

        initial_epochs = 200

        loss0, accuracy0 = self.model.evaluate(val_batches)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        try:
            history = self.model.fit(train_batches,
                                     epochs=initial_epochs,
                                     validation_data=val_batches,
                                     callbacks=self.callbacks,
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

        train = raw_train.map(self.format_fn)
        val = raw_val.map(self.format_fn)

        train_loss, train_accuracy = self.model.evaluate(train)
        val_loss, val_accuracy = self.model.evaluate(val)

        print("train loss: {:.2f}".format(train_loss))
        print("train accuracy: {:.2f}".format(train_accuracy))
        print("val loss: {:.2f}".format(val_loss))
        print("val accuracy: {:.2f}".format(val_accuracy))

    def predict(self, np_image):
        formatted_image = self.format_fn(np_image, "?")

        predictions = self.model.predict(tf.expand_dims(formatted_image[0], axis=0))
        if self.base_model_only:
            label = decode_predictions(predictions)
        else:
            label = self.label_fn(predictions)
        version = self.version

        prob_list = []
        for prob in predictions:
            prob_list.append(prob.astype(float))
        return prob_list, label, version

    def save(self):
        with open("model.json", "w") as json_file:
            model_json = self.model.to_json()
            json_file.write(model_json)
        # self.model.save_weights('final_model', save_format='tf')


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


def format_example(image, label):
    # image is already in range 0 ~ +1 -> change it to -1 ~ +1
    image = (image * 2) - 1
    return image, label


def print_min_max(image):
    print("min", tf.math.reduce_min(image))
    print("max", tf.math.reduce_max(image))


def load_model(base_model_name: str, class_names: typing.List) -> DeepModel:
    img_size = 224
    img_shape = (img_size, img_size, 3)
    if base_model_name == "VGG16":
        base_model = keras.applications.VGG16(input_shape=img_shape,
                                              include_top=False,
                                              weights='imagenet')
        format_fn = format_example
        base_model_only = False
    elif base_model_name == "MobileNetV2":
        base_model = keras.applications.MobileNetV2(input_shape=img_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        format_fn = format_example
        base_model_only = False
    elif base_model_name == "Xception":
        img_size = 299
        img_shape = (img_size, img_size, 3)
        base_model = keras.applications.Xception(input_shape=img_shape,
                                                 include_top=False,
                                                 weights='imagenet')
        format_fn = format_example
        base_model_only = False
    elif base_model_name == "MobileNetV2_only":
        base_model = keras.applications.MobileNetV2(input_shape=img_shape,
                                                    include_top=True,
                                                    weights='imagenet')
        format_fn = format_example
        base_model_only = True
    elif base_model_name == "Xception_only":
        img_size = 299
        img_shape = (img_size, img_size, 3)
        base_model = keras.applications.Xception(input_shape=img_shape,
                                                 include_top=True,
                                                 weights='imagenet')
        format_fn = format_example
        base_model_only = True
    else:
        raise ValueError("Unknown base model {}".format(base_model_name))

    return DeepModel(base_model, format_fn, base_model_only, base_model_name, simple_label, class_names)


def simple_label(prediction):
    if np.argmax(prediction).astype('int32') == 0:
        return "True"
    else:
        return "False"
