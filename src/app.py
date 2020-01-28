import json
import os

from flask import Flask
from flask import request

from serving import model_loader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(base_path)
model = model_loader.load_model(base_path)


@app.route('/')
def index():
    return 'Hello Flask'


@app.route('/pokemon_classification', methods=['POST'])
def pokemon_classification():
    # requested_url 으로 전달된 url 이미지를 다운 받고 분석
    json_request = request.json

    filepath = model.save_img(json_request)
    result = model.predict(str(filepath), "pokemon_model")
    print(result)
    return json.dumps(result), 200


@app.route('/people_classification', methods=['POST'])
def people_classification():
    # requested_url 으로 전달된 url 이미지를 다운 받고 분석
    json_request = request.json

    filepath = model.save_img(json_request)
    result = model.predict(str(filepath), "people_model")
    return json.dumps(result), 200
