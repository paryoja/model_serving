import base64
import hashlib
import json
import multiprocessing
import pathlib
import shutil

import lz4.frame
import requests
import tqdm


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


def parse(data):
    url = data['fields']['url']
    url_hash = int(hashlib.sha1(url.encode('utf-8')).hexdigest(), 16) % 100
    count = 0
    if url_hash < 90:
        save_image(url, 'data/train/' + str(data['fields']['selected']), url.split('/')[-1])
    else:
        save_image(url, 'data/validate/' + str(data['fields']['selected']), url.split('/')[-1])
    count += 1


def download(label="True"):
    page = 1
    with multiprocessing.Pool(10) as pool:
        while True:
            results = requests.get("http://13.125.1.208/book/people_result/download/" + label + "/" + str(page))

            pickled = base64.b85decode(results.text)

            decompressed = lz4.frame.decompress(pickled)

            with open('../zip.txt', 'w') as w:
                w.write(results.text)

            data = decompressed.decode('utf-8')
            with open('../result.txt', 'w') as w:
                w.write(data)

            data_json = json.loads(data)
            image_list = data_json["image_list"]

            with tqdm.tqdm(total=len(image_list)) as pbar:
                for i, _ in enumerate(pool.imap_unordered(parse, image_list)):
                    pbar.update()
            page += 1
            if not data_json["has_next"]:
                break
