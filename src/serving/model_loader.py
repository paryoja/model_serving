import datetime
import json
import pathlib
import shutil

import requests
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from model import keras_model
from train import data_loader


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


class ServingModel:
    def __init__(self, deep_model_map):
        self.deep_model_map = deep_model_map
        self.media_root = "./media/"

    def save_img(self, json_request):
        current_date = datetime.datetime.now().strftime('%Y_%m_%d')
        filepath = download_image(json_request["requested_url"], self.media_root + current_date)
        return filepath

    def predict(self, filepath, model_name):
        deep_model = self.deep_model_map[model_name]
        img = tf.io.read_file(filepath)
        img = data_loader.decode_img(img, deep_model["model_info"].data_info.img_size)
        classification, label, version = deep_model["deep_model"].predict(img)

        result = {
            'status': 'success',
            'classification': classification,
            'label': label,
            'version': version,
        }
        return result


def load_model(base_path):
    with open(base_path + "/model_info.json") as f:
        all_model_info = json.load(f)

    deep_model_map = {}
    for k, v in all_model_info.items():
        model_path = pathlib.Path(base_path + "/" + v)
        json_model = model_path.parents[0] / 'model.json'
        with open(json_model) as f:
            loaded_model = model_from_json(f.read())
        loaded_model.load_weights(base_path + "/" + v)

        model_info_json = model_path.parents[0] / 'model_info.json'
        model_info = keras_model.ModelInfo.load(str(model_info_json))

        deep_model = keras_model.LoadedModel(model_info, loaded_model)
        deep_model_map[k] = {'deep_model': deep_model, 'model_info': model_info}
    print(deep_model_map)
    return ServingModel(deep_model_map)
