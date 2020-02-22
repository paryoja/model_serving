import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from train import data_downloader


class ModelType:
    Xception = "Xception"
    MobileNetV2 = "MobileNetV2"
    VGG16 = "VGG16"
    NASNetMobile = "NASNetMobile"
    InceptionResNetV2 = "InceptionResNetV2"

    img_size_map = {
        Xception: 299,
        MobileNetV2: 224,
        VGG16: 224,
        NASNetMobile: 224,
        InceptionResNetV2: 299,
    }

    def __init__(self, model_str):
        self.model_str = model_str

    @property
    def img_size(self):
        return self.img_size_map[self.model_str]


def main(need_download=False):
    from model import keras_model
    from model import custom_model
    from model import model_info
    from plot import show_single_image
    from train import data_loader

    import matplotlib.pyplot as plt
    plt.interactive(False)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #     except RuntimeError as e:
    #         # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    #         print(e)

    data_type = data_loader.DataType(data_loader.DataType.People)

    # 다운 시작
    if need_download:
        session = data_downloader.get_session()
        # data 폴더에 train/validate 폴더를 나누고 label 이름의 폴더로 다운로드 진행
        data_type.download(session)

    # load data
    model_type = ModelType(ModelType.NASNetMobile)
    train_model = True
    base_model_only = False

    data_info = data_loader.DatasetInfo(img_size=model_type.img_size, data_type=data_type)
    dataset = data_loader.Dataset(data_info)

    model_info = model_info.ModelInfo(model_type.model_str, base_model_only=base_model_only,
                                      model_name=data_type.data_str, class_names=dataset.class_names,
                                      version=model_type.model_str + "_1.0", data_info=data_info,
                                      img_size=model_type.img_size, load_model=True,
                                      load_model_path="models/People/NASNetMobile_1.0/2020_02_22_12_32_07/model.022-1.02.hdf5")

    # model_info = model_info.ModelInfo(model_type.model_str, base_model_only=base_model_only,
    #                                   model_name=data_type.data_str, class_names=dataset.class_names,
    #                                   version=model_type.model_str + "_1.0", data_info=data_info,
    #                                   img_size=model_type.img_size)

    model = custom_model.CNNWithDense(model_info)
    data_loader.confusing_data(model, data_info)

    if train_model:
        # CHECK which file is not trained
        untrained_dataset = data_loader.Dataset(data_info, from_untrained_file=True)

        last_layer_train_info = keras_model.TrainInfo(learning_rate=1e-5, momentum=0.9, update_base=False,
                                                      warmup_batches=15,
                                                      epochs=30)
        model.train_model(untrained_dataset, last_layer_train_info)
        model.train_model(dataset, last_layer_train_info)

        all_layer_train_info = keras_model.TrainInfo(learning_rate=1e-6, momentum=0.9, update_base=True,
                                                     warmup_batches=15,
                                                     epochs=30)
        model.train_model(dataset, all_layer_train_info)

    # sample train and test with it
    train, val = dataset.get_raw_data()
    for image in train:
        prob_list, label, version, class_names = model.predict(image[0])
        show_single_image(image, label)

        print(label)
        input("Next")


if __name__ == "__main__":
    main(need_download=True)
