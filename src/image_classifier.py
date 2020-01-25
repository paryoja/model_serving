import copy
import pathlib
import shutil

import numpy as np
import requests
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import vgg16, VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


class Classifier:
    def __init__(self):
        print("Loading models")
        self.num_labels = 2

        base_model = VGG16(include_top=False,
                           input_tensor=Input(shape=(224, 224, 3)))

        # construct the head of the model that will be placed on top of the
        # the base model
        head_model = base_model.output
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(512, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(self.num_labels, activation="softmax")(head_model)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model_file = "model (1).h5"
        self.model = Model(inputs=base_model.input, outputs=head_model)
        self.model.load_weights("/work/models/{}".format(model_file))
        self.version = model_file

        self.label_map = {
            0: 'True',
            1: 'False'
        }

    def predict(self, data):
        return self.model.predict(data)[0]

    def get_label(self, prediction):
        return self.label_map[np.argmax(prediction).astype('int32')]


model = Classifier()


def download_image(url, directory):
    filename = pathlib.Path(url).name
    directory_path = pathlib.Path(directory)
    filepath = directory_path / filename

    if not filepath.exists():

        r = requests.get(url, stream=True)

        if r.status_code == 200:
            directory_path.mkdir(exist_ok=True, parents=True)

            with open(directory_path / filename, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
    return filepath


def classification(filepath):
    img = load_img(filepath, target_size=(224, 224))
    numpy_image = img_to_array(img)

    np_image = vgg16.preprocess_input(np.expand_dims(numpy_image, 0))

    predictions = model.predict(np_image)
    label = model.get_label(predictions)
    version = model.version

    prob_list = []
    for prob in predictions:
        prob_list.append(prob.astype(float))
    return prob_list, label, version


def restore_original_image_from_array(x, data_format='channels_first'):
    mean = [103.939, 116.779, 123.68]

    x = copy.deepcopy(x)
    # Zero-center by mean pixel

    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]

    # 'BGR'->'RGB'
    x = x[..., ::-1]

    return x
