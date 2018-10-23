import luigi
from keras.engine.training import Model
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras import optimizers, applications
from keras.models import Sequential

from pneumonia.model.base import BaseKerasModel
from pneumonia.data.input import create_train_generator, create_val_generator
from pneumonia.data.preparation import BalanceDataset

class INCEPTIONV3(BaseKerasModel):

    activation = luigi.Parameter(default="relu")
    input_shape = luigi.TupleParameter(default=(224, 224))
    optimizer = luigi.Parameter(default="adam")
    pooling = luigi.Parameter(default="avg")
    dropout = luigi.FloatParameter(default=0.3)
    learning_rate = luigi.FloatParameter(default=0.0001)
    dense_neurons = luigi.IntParameter(default=1024)
    categorizator_epochs = luigi.IntParameter(default=15)
    
    def requires(self):
        return BalanceDataset(sampling_strategy="oversample")

    def create_model(self) -> Model:
        base_model = applications.InceptionV3(weights='imagenet', include_top=False, pooling=self.pooling,
                                   input_shape=(self.input_shape[0], self.input_shape[1], 3))
        
        x = base_model.output
        x = Dense(self.dense_neurons, activation=self.activation)(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.dense_neurons, activation=self.activation)(x)
        x = Dropout(self.dropout)(x)
        categorizator = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=categorizator)

        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        print("Training the categorizator")

        categorizator_train_generator = self.get_train_generator()
        categorizator_val_generator = self.get_val_generator()

        model.fit_generator(categorizator_train_generator, epochs=self.categorizator_epochs,
                                steps_per_epoch=(len(categorizator_train_generator.filenames) / self.batch_size),
                                validation_data=categorizator_val_generator,
                                validation_steps=(len(categorizator_val_generator.filenames) / self.val_batch_size),
                                verbose=1,
                                workers=self.generator_workers,
                                max_queue_size=self.generator_max_queue_size,
                                use_multiprocessing=True)

        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

        optimizer = optimizers.get({'class_name': self.optimizer, 'config': {"lr": self.learning_rate}})
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
