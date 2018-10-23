import luigi
from keras.engine.training import Model
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras import optimizers, applications
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D

from pneumonia.model.base import BaseKerasModel
from pneumonia.data.preparation import BalanceDataset

class DENSENET121(BaseKerasModel):

    input_shape = luigi.TupleParameter(default=(224, 224))
    frozen_layers = luigi.IntParameter(default=0)
    dense_layers = luigi.IntParameter(default=2)
    optimizer = luigi.Parameter(default="adam")
    dropout = luigi.FloatParameter(default=0.35)
    momentum = luigi.FloatParameter(default=0.6)
    nesterov = luigi.BoolParameter(default=False)
    activation_func = luigi.Parameter(default="relu")
    final_activation = luigi.Parameter(default="sigmoid")
    dense_neurons = luigi.IntParameter(default=1024)
    learning_rate = luigi.FloatParameter(default=0.0001)
    pooling_strategy = luigi.Parameter(default="avg")
    kernel_initializer = luigi.ChoiceParameter(choices=["glorot_uniform", "lecun_normal", "he_uniform"], default="glorot_uniform")
    has_dropout_between_dense = luigi.BoolParameter(default=True)
    

    def create_model(self) -> Model:
        base_model = applications.DenseNet121(weights='imagenet', include_top=False, 
                                   pooling=self.pooling_strategy,
                                   input_shape=(self.input_shape[0], self.input_shape[1], 3))
        
        for layer in base_model.layers[:self.frozen_layers]:
            layer.trainable = False
        
        model = Sequential()
        model.add(base_model)
        if self.pooling_strategy == None:
            model.add(Flatten())
            
        for _ in range(self.dense_layers):
            model.add(Dense(self.dense_neurons, activation=self.activation_func,                                    kernel_initializer=self.kernel_initializer))
            if self.dropout:
                model.add(Dropout(self.dropout))
   
        model.add(Dense(1, activation=self.final_activation))

        optimizer = optimizers.get({'class_name': self.optimizer, 'config': {"lr": self.learning_rate}})
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
