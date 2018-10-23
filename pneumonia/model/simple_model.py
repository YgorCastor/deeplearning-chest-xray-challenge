import luigi
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.noise import AlphaDropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

from pneumonia.model.base import BaseKerasModel


class SimpleModel(BaseKerasModel):

    learning_rate = luigi.FloatParameter(default=0.0005)

    def create_model(self) -> Model:
        model = Sequential()

        model.add(Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=(self.input_shape[0], self.input_shape[1], 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(512, kernel_initializer='lecun_normal', activation='selu'))
        model.add(AlphaDropout(0.3))
        model.add(Dense(512, kernel_initializer='lecun_normal', activation='selu'))
        model.add(AlphaDropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model