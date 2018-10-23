import luigi
from keras.engine.training import Model
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras import optimizers, applications
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D

from pneumonia.model.base import BaseKerasModel
from pneumonia.data.preparation import BalanceDataset

class INCEPTIONRESNET(BaseKerasModel):

    input_shape = luigi.TupleParameter(default=(224, 224))
    frozen_layers = luigi.IntParameter(default=10)
    optimizer = luigi.Parameter(default="adam")
    dropout = luigi.FloatParameter(default=0.3)
    momentum = luigi.FloatParameter(default=0.8)
    activation_func = luigi.Parameter(default="selu")
    final_activation = luigi.Parameter(default="sigmoid")
    dense_neurons = luigi.IntParameter(default=2048)
    learning_rate = luigi.FloatParameter(default=0.0001)
    global_average_pooling = luigi.BoolParameter(default=False)
    
    
    def requires(self):
        return BalanceDataset(sampling_strategy="oversample")

    def create_model(self) -> Model:
        base_model = applications.InceptionResNetV2(weights='imagenet', include_top=False,
                                   input_shape=(self.input_shape[0], self.input_shape[1], 3))
        for layer in base_model.layers[:self.frozen_layers]:
            layer.trainable = False
        
        model = Sequential()
        model.add(base_model)
        if self.global_average_pooling:
            model.add(GlobalAveragePooling2D())
        else:
            model.add(Flatten())
        model.add(Dense(self.dense_neurons, activation=self.activation_func))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation=self.final_activation))

        optimizer = optimizers.get({'class_name': self.optimizer, 'config': {"lr": self.learning_rate}})
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
