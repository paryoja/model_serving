import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.models import model_from_json

from model import model_info as model_info_file
from train import augment_util
from train import train_util


class TrainInfo:
    def __init__(self, learning_rate=0.0001, momentum=0.9, update_base=False, shuffle_buffer_size=1000, batch_size=92,
                 warmup_batches=10, epochs=100):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.update_base = update_base
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.warmup_batches = warmup_batches
        self.epochs = epochs


def plot_5by5(raw_train, class_names):
    plt.figure(figsize=(10, 10))
    for (image, label), i in zip(raw_train, range(25)):
        plt.subplot(5, 5, i + 1)
        plt.title(class_names[tf.argmax(tf.cast(label, tf.int32))])
        plt.imshow(image)
        plt.axis('off')
    plt.show()


def print_min_max(image):
    print("min", tf.math.reduce_min(image))
    print("max", tf.math.reduce_max(image))


class DeepModel:
    def __init__(self, model_info: model_info_file.ModelInfo, *args):

        self.model_info = model_info

        if not self.model_info.base_model_only:
            self.model = self.make_model(*args)
        else:
            self.model = self.model_info.base_model

    def make_model(self, *args):
        raise NotImplementedError()

    def make_callbacks(self, train_info: TrainInfo):
        cb_lrate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1,
                                                     mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-7)
        cb_tb_hist = keras.callbacks.TensorBoard(log_dir=str(self.model_info.graph_dir), histogram_freq=0,
                                                 write_graph=True, write_images=False, profile_batch=None)
        cb_checkpoint = keras.callbacks.ModelCheckpoint(
            str(self.model_info.model_dir / './model.{epoch:03d}-{val_loss:.2f}.hdf5'), verbose=1,
            monitor='val_accuracy',
            save_best_only=True, mode='auto')
        warm_up_lr = train_util.WarmUpLearningRateScheduler(train_info.warmup_batches, init_lr=train_info.learning_rate,
                                                            verbose=1)

        return [cb_lrate, cb_tb_hist, cb_checkpoint, warm_up_lr]

    def train_model(self, dataset, train_info: TrainInfo):
        print("Starting Train")
        raw_train, raw_val = dataset.get_raw_data()

        raw_train = raw_train.cache()
        raw_val = raw_val.cache()

        plot_5by5(raw_train, self.model_info.class_names)
        train = augment_util.get_augmented_dataset(raw_train, self.model_info)

        train = train.map(self.model_info.format_fn)
        val = raw_val.map(self.model_info.format_fn)

        self.model_info.base_model.trainable = train_info.update_base

        # 일단 여기서 cache 해본다 -> 메모리 부족시 앞에서 해본다
        train_batches = train.shuffle(train_info.shuffle_buffer_size).prefetch(10000)

        train_batches = train_batches.batch(train_info.batch_size)
        val_batches = val.cache().batch(train_info.batch_size)

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(lr=train_info.learning_rate, momentum=train_info.momentum),
            loss="categorical_crossentropy",
            metrics=['accuracy'])
        self.save()
        self.model.summary()

        initial_epochs = train_info.epochs

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
        class_names = {}
        for idx, class_name in enumerate(self.model_info.class_names):
            class_names[class_name.astype(str)] = idx
        return prob_list, label, version, class_names

    def predict_batch(self, images):
        return self.model.predict(images)

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
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    plt.savefig('accuracy_{}.png'.format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
