import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from train_main import ModelType


def main(need_download=False):
    from train import data_loader
    from serving import model_loader

    import matplotlib.pyplot as plt
    plt.interactive(False)

    data_type = data_loader.DataType(data_loader.DataType.People)

    # load data
    model_type = ModelType(ModelType.NASNetMobile)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_path)
    model = model_loader.load_model(base_path)

    data_info = data_loader.DatasetInfo(img_size=model_type.img_size, data_type=data_type)
    data_loader.validation(data_info)

    data_loader.confusing_data(model, data_info)


if __name__ == "__main__":
    main(True)
