import luigi
from keras.engine.training import Model
from keras.layers import Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras import optimizers, applications
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

from pneumonia.model.base import TransferLearningKerasModel


class VGG19(TransferLearningKerasModel):

    input_shape = luigi.TupleParameter(default=(256, 256))
    frozen_layers = luigi.IntParameter(default=5)
    dense_neurons = luigi.IntParameter(default=1024)
    dropout = luigi.FloatParameter(default=None)
    optimizer = luigi.Parameter(default="adam")
    learning_rate = luigi.FloatParameter(default=0.0001)
    kernel_initializer = luigi.Parameter(default="glorot_uniform")
    total_dense_layers = luigi.IntParameter(default=1)
    all_dropout = luigi.BoolParameter(default=True)
    batch_normalization_after_vgg = luigi.BoolParameter(default=False)
    batch_normalization_between_dense = luigi.BoolParameter(default=False)
     
    def create_base_model(self) -> Model:
        return applications.VGG19(weights="imagenet", include_top=False, pooling='avg', input_shape=(self.input_shape[0], self.input_shape[1], 3))

    def create_model_with(self, base_model: Model) -> Model:
        model = Sequential()
        model.add(self.base_model)
        # model.add(Flatten())
        
        if self.batch_normalization_after_vgg:
            model.add(BatchNormalization())

        cont = 1
        while cont <= self.total_dense_layers:
            model.add(Dense(self.dense_neurons, activation="relu", kernel_initializer=self.kernel_initializer))
            if self.dropout != None and self.all_dropout:
                model.add(AlphaDropout(self.dropout))
                
            if self.batch_normalization_between_dense:
                model.add(BatchNormalization())
                
            cont = cont + 1
            
        if self.dropout != None and not self.all_dropout:
            model.add(AlphaDropout(self.dropout))
            
        model.add(Dense(1, activation='sigmoid', kernel_initializer=self.kernel_initializer))
        
        optimizer = optimizers.get({'class_name': self.optimizer, 'config': {"lr": self.learning_rate}})
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
