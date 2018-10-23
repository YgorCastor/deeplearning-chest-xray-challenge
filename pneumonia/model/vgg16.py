import luigi
from keras.engine.training import Model
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras import applications
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from pneumonia.model.base import TransferLearningKerasModel


class VGG16(TransferLearningKerasModel):
    input_shape = luigi.TupleParameter(default=(224, 224))
    dense_neurons = luigi.IntParameter(default=4096)
    dense_layers = luigi.IntParameter(default=2)
    dropout = luigi.FloatParameter(default=None)
    dropout_after_vgg16 = luigi.FloatParameter(default=None)
    activation_function = luigi.ChoiceParameter(choices=["relu", "selu"], default="relu")
    kernel_initializer = luigi.ChoiceParameter(choices=["glorot_uniform", "lecun_normal"], default="glorot_uniform")
    batch_normalization_after_vgg16 = luigi.BoolParameter(default=False)
    batch_normalization_between_dense = luigi.BoolParameter(default=False)

    def create_base_model(self) -> Model:
        return applications.VGG16(weights="imagenet", include_top=False, pooling='avg',
                                        input_shape=(self.input_shape[0], self.input_shape[1], 3))

    def create_model_with(self, base_model: Model) -> Model:
        model = Sequential()
        model.add(self.base_model)

        if self.batch_normalization_after_vgg16:
            model.add(BatchNormalization())
        if self.dropout_after_vgg16:
            model.add(Dropout(self.dropout_after_vgg16))

        # model.add(Flatten())
        for _ in range(self.dense_layers):
            model.add(
                Dense(self.dense_neurons, activation=self.activation_function, kernel_initializer=self.kernel_initializer))
            if self.batch_normalization_between_dense:
                model.add(BatchNormalization())
            if self.dropout:
                model.add(Dropout(self.dropout))

        model.add(Dense(1, activation='sigmoid'))

        return model

