import os

from train import data_downloader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    from train import data_loader
    from model import keras_model
    from model import custom_model
    from model import model_info
    from plot import show_single_image

    import matplotlib.pyplot as plt
    plt.interactive(False)

    data_type = data_loader.DataType(data_loader.DataType.PokemonYesNo)

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
                                      img_size=model_type.img_size)
    model = custom_model.CNNWithDense(model_info)

    # model.train(dataset, update_base=True)
    if train_model:
        raw, _ = dataset.get_raw_data()
        print(id(raw))
        for img, label in raw:
            plt.figure()
            plt.imshow(img)
            plt.title('start')
            plt.show()
            break

        train_info = keras_model.TrainInfo(learning_rate=1e-5, momentum=0.9, update_base=False, warmup_batches=15,
                                           epochs=30)
        model.train_model(dataset, train_info)

        train_info = keras_model.TrainInfo(learning_rate=1e-6, momentum=0.9, update_base=True, warmup_batches=15,
                                           epochs=30)
        model.train_model(dataset, train_info)

    # sample train and test with it
    train, val = dataset.get_raw_data()
    for image in train:
        prob_list, label, version, class_names = model.predict(image[0])
        show_single_image(image, label)

        print(label)
        input("Next")


if __name__ == "__main__":
    main(False)
