from tensorflow import keras

from . import keras_model


class CNNWithDense(keras_model.DeepModel):
    def __init__(self, model_info):
        super(CNNWithDense, self).__init__(model_info)

    def make_model(self):
        additional_conv_layer = keras.layers.Conv2D(
            32, 3, padding="same", activation="relu"
        )
        dropout_layer_1 = keras.layers.Dropout(0.5)
        global_average_layer = keras.layers.GlobalAveragePooling2D()
        dense_layer = keras.layers.Dense(512, activation="relu")
        dropout_layer_2 = keras.layers.Dropout(0.5)
        prediction_layer = keras.layers.Dense(
            len(self.model_info.class_names), activation="softmax"
        )

        model = keras.Sequential(
            [
                self.model_info.base_model,
                additional_conv_layer,
                dropout_layer_1,
                global_average_layer,
                dense_layer,
                dropout_layer_2,
                prediction_layer,
            ]
        )

        return model
