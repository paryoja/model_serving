import os

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img_size_map = {
    "Xception": 299,
    "MobileNetV2": 224,
    "VGG16": 224,
    "NASNetMobile": 224,
}


def main(need_download=False):
    from train import data_downloader
    from train import data_loader
    from model import keras_model
    from model import custom_model
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
    model_name = "VGG16"
    # model_name = "MobileNetV2"
    # model_name = "Xception"
    base_model_only = False

    data_info = data_loader.DatasetInfo(img_size=img_size_map[model_name])
    dataset = data_loader.Dataset(data_info)

    model_info = keras_model.ModelInfo(model_name, base_model_only=base_model_only, model_name="pokemon_model",
                                       class_names=dataset.class_names, version=model_name + "_1.0",
                                       data_info=data_info)
    model = custom_model.CNNWithDense(model_info)

    # model.train(dataset, update_base=True)
    if "only" not in model_name:
        train_info = keras_model.TrainInfo(learning_rate=1e-3, momentum=0.9, update_base=False)
        model.train_model(dataset, train_info)

    # sample train and test with it
    train, val = dataset.get_raw_data()
    for image in train:
        prob_list, label, version = model.predict(image[0])
        show_single_image(image, label)

        print(label)
        input("Next")


if __name__ == "__main__":
    main(False)
