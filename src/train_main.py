import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(need_download=False):
    from train import data_downloader
    from train import data_loader
    from model import keras_model
    from plot import show_single_image
    plt.interactive(False)

    # 다운 시작
    if need_download:
        # data 폴더에 train/validate 폴더를 나누고 label 이름의 폴더로 다운로드 진행
        data_downloader.download_pokemon(label="yes")
        data_downloader.download_pokemon(label="no")
        data_downloader.download_pokemon(label="more")
        data_downloader.download_pokemon(label="little")
        data_downloader.validate_image()
    # load data
    # model_name = "MobileNetV2"
    model_name = "Xception"
    dataset = data_loader.Dataset(model_name)

    # load model
    # model = keras_model.load_model("MobileNetV2_only")
    model = keras_model.load_model(model_name, dataset.class_names)

    # model.train(dataset, update_base=True)
    if "only" not in model_name:
        model.train(dataset, update_base=False)

        # model.load_file("model.h5", "model.json")

    train, val = dataset.get_raw_data()
    for image in train:
        prob_list, label, version = model.predict(image[0])
        show_single_image(image, label)

        print(label)
        input("Next")


if __name__ == "__main__":
    main(True)
