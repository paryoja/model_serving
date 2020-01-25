import datetime
import json

from flask import Flask
from flask import request

import image_classifier

app = Flask(__name__)

media_root = './media/'


@app.route('/')
def index():
    return 'Hello Flask'


@app.route('/people_classification', methods=['POST'])
def people_classification():
    # requested_url 으로 전달된 url 이미지를 다운 받고 분석
    json_request = request.json

    current_date = datetime.datetime.now().strftime('%Y_%m_%d')
    filepath = image_classifier.download_image(json_request["requested_url"], media_root + current_date)
    classification, label, version = image_classifier.classification(filepath)

    result = {
        'status': 'success',
        'classification': classification,
        'label': label,
        'version': version,
    }

    return json.dumps(result), 200
