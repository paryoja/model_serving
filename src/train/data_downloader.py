import base64
import functools
import hashlib
import json
import multiprocessing
import pathlib
import shutil

import lz4.frame
import requests
import tensorflow as tf
import tqdm
from PIL import Image


def save_image(url, directory, filename):
    directory_path = pathlib.Path(directory)
    file_path = directory_path / filename

    if file_path.exists():
        return

    directory_path.mkdir(exist_ok=True, parents=True)

    try:
        r = requests.get(url, stream=True)
    except requests.exceptions.ConnectionError:
        print("Skipping {}".format(url))
        return

    if r.status_code == 200:
        with open(file_path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


def parse(data, data_type="pokemon_yes_no", black_list=None):
    url = data['fields']['url']
    filename = url.split('/')[-1]

    url_hash = int(hashlib.sha1(url.encode('utf-8')).hexdigest(), 16) % 100

    if data_type == "pokemon_yes_no":
        label = data['fields']['classified']
    elif data_type == "pokemon_classification":
        label = data['fields']['original_label']
    else:
        label = data['fields']['selected']

    if url_hash < 90:
        target_path = 'data/train/' + str(label)
    else:
        target_path = 'data/validate/' + str(label)

    # print(black_list)
    # print(target_path + "/" + filename)
    if black_list and target_path + "/" + filename in black_list:
        print("Skipping {} since it is listed in blacklist".format(url))
    save_image(url, target_path, filename)


def validate_image():
    print("Validate Images")
    if pathlib.Path('blacklist.json').exists():
        with open('blacklist.json', 'r') as f:
            ignore_list = json.load(f)
    else:
        ignore_list = []
    for file_path in pathlib.Path("data").glob("**/*"):
        if file_path.is_file():
            str_file_path = str(file_path)

            img = tf.io.read_file(str_file_path)
            try:
                tf.image.decode_jpeg(img, channels=3)
            except Exception:
                print("Converting", str_file_path)
                try:
                    im = Image.open(str_file_path)
                    im.save(str_file_path, "JPEG")

                except Exception:
                    print("Converting Failed add to ignore list")
                    file_path.unlink()
                    str_file_path = str_file_path.replace("\\", "/")  # normalize directory splitter
                    if str_file_path not in ignore_list:
                        ignore_list.append(str_file_path)

    with open('blacklist.json', 'w') as w:
        w.write(json.dumps(ignore_list))


def download_pokemon(label="yes"):
    download(url="http://13.125.1.208/book/pokemon_export/", label=label)


def download(url, label="True", data_type="pokemon_yes_no"):
    page = 1

    black_list_path = pathlib.Path("blacklist.json")
    black_list = None
    if black_list_path.exists():
        with open(black_list_path) as f:
            black_list = json.load(f)
        black_list = set(black_list)

    parse_function = functools.partial(parse, data_type=data_type, black_list=black_list)
    with multiprocessing.Pool(10) as pool:
        while True:
            request_url = url + label + "/" + str(page)
            results = requests.get(url + label + "/" + str(page))
            print(request_url)

            pickled = base64.b85decode(results.text)

            decompressed = lz4.frame.decompress(pickled)

            with open('./zip.txt', 'w') as w:
                w.write(results.text)

            data = decompressed.decode('utf-8')
            with open('./result.txt', 'w') as w:
                w.write(data)

            data_json = json.loads(data)
            image_list = data_json["image_list"]

            with tqdm.tqdm(total=len(image_list)) as pbar:
                for i, _ in enumerate(pool.imap_unordered(parse_function, image_list)):
                    pbar.update()
            page += 1
            if not data_json["has_next"]:
                break
