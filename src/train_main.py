import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(need_download=False):
    from train import data_downloader
    from train import data_loader
    from model import keras_model

    plt.interactive(False)

    # 다운 시작
    if need_download:
        # data 폴더에 train/validate 폴더를 나누고 label 이름의 폴더로 다운로드 진행
        data_downloader.download(label="True")
        data_downloader.download(label="False")

    # load data
    dataset = data_loader.Dataset()

    # load model
    model = keras_model.load_model("MobileNetV2")

    # model.train(dataset, update_base=True)
    model.train(dataset, update_base=False)


if __name__ == "__main__":
    main(True)
