import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf

from train import data_downloader
from train import data_loader
from utils.model_logger import logger


def main():
    data_type = data_loader.DataType(data_loader.DataType.People)
    data_info = data_loader.DatasetInfo(img_size=600, data_type=data_type)
    untrained_dataset = data_loader.Dataset(data_info, keep_path=True, from_untrained_file=True)

    session = data_downloader.get_session()
    rating_url = 'http://13.125.1.208/book/rating/api'
    get_id_url = 'http://13.125.1.208/book/people/get_id'

    revised_file = 'revised.txt'
    revised_path = pathlib.Path(revised_file)
    revised = set()
    if revised_path.exists():
        with open(revised_path) as f:
            for line in f:
                revised.add(pathlib.Path(line.strip()))

    try:
        for (image, path), label in untrained_dataset.get_raw_data()[0]:
            plt.title(untrained_dataset.class_names[tf.argmax(tf.cast(label, tf.int32))])
            plt.imshow(image)
            plt.axis('off')
            plt.show()

            query_file = pathlib.Path(path.numpy().decode('utf-8'))
            if query_file in revised:
                continue
            query = query_file.name
            result = session.post(get_id_url,
                                  data={'query': query, 'csrfmiddlewaretoken': session.cookies.get("csrftoken")})
            logger.debug(query)
            print(result.text)

            image_id = result.text
            data = {
                'image_id': image_id,
                'csrfmiddlewaretoken': session.cookies.get("csrftoken"),
                'data_type': 'people'
            }

            new_label = input("{} y/n: ".format(path))

            if new_label == "y":
                data['selected'] = 'Yes'
                result = session.post(rating_url, data=data)
                print(result)
                print(result.text)
            elif new_label == "n":
                data['selected'] = 'No'
                result = session.post(rating_url, data=data)
                print(result)
                print(result.text)
            elif new_label == "s":
                break
            revised.add(query_file)
    finally:
        with open(revised_path, 'w') as f:
            for revised_item in revised:
                f.write(str(revised_item))
                f.write('\n')


if __name__ == "__main__":
    main()
