import luigi
from keras.engine.training import Model
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras import optimizers, applications
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D

from pneumonia.model.base import BaseKerasModel
from pneumonia.data.preparation import BalanceDataset

class DENSENET201(BaseKerasModel):

    input_shape = luigi.TupleParameter(default=(224, 224))
    frozen_layers = luigi.IntParameter(default=0)
    optimizer = luigi.Parameter(default="nadam")
    dropout = luigi.FloatParameter(default=0.5)
    momentum = luigi.FloatParameter(default=0.8)
    activation_func = luigi.Parameter(default="selu")
    dense_neurons = luigi.IntParameter(default=2048)
    learning_rate = luigi.FloatParameter(default=0.00001)
    pooling_strategy = luigi.Parameter(default="avg")
    nesterov_momentum = luigi.BoolParameter(default=False)   
    has_dropout_between_dense = luigi.BoolParameter(default=False)
    
    def requires(self):
        return BalanceDataset(sampling_strategy="oversample")

    def create_model(self) -> Model:
        base_model = applications.DenseNet201(weights='imagenet', include_top=False, 
                                   pooling=self.pooling_strategy,
                                   input_shape=(self.input_shape[0], self.input_shape[1], 3))
        
        for layer in base_model.layers[:self.frozen_layers]:
            layer.trainable = False
        
        model = Sequential()
        model.add(base_model)
 
        model.add(Dense(self.dense_neurons, activation=self.activation_func))
        if self.has_dropout_between_dense:
            model.add(Dropout(self.dropout))
                
        model.add(Dense(self.dense_neurons, activation=self.activation_func))
        if self.has_dropout_between_dense:
            model.add(Dropout(self.dropout))

        model.add(Dense(1, activation="sigmoid"))
        
        if self.optimizer == "SGD":
            optimizer = optimizers.get({'class_name': self.optimizer, 
                                        'config': {"lr": self.learning_rate, "nesterov" : self.nesterov_momentum,
                                        'momentum': self.momentum}})
        else:
            optimizer = optimizers.get({'class_name': self.optimizer, 'config': {"lr": self.learning_rate}})
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model