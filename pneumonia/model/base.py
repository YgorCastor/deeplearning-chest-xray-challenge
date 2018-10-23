from typing import List, Type

import luigi
import abc
import json
from contextlib import redirect_stdout
import os
import shutil
import multiprocessing
import time

from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.training import Model

from keras.preprocessing.image import DirectoryIterator

from pneumonia.data.input import create_train_generator, create_val_generator, create_test_generator
from pneumonia.data.preparation import BalanceDataset, ResizeAndAugmentImages
from pneumonia.files import get_params_path, get_weights_path, get_training_result_path, get_params, \
    get_train_confusion_matrix_path, get_val_confusion_matrix_path, get_training_log_path
from pneumonia.evaluation.commons import TrainingResult, EvaluationReport
from pneumonia.evaluation.keras import evaluate_keras_model

from pneumonia.plot import plot_training_log, plot_confusion_matrix


class BaseModel(luigi.Task):
    __metaclass__ = abc.ABCMeta

    val_size = luigi.FloatParameter(default=0.2)
    invert_train_val = luigi.BoolParameter(default=False)
    prior_augmentation = luigi.BoolParameter(default=False)
    prior_augmentation_times = luigi.IntParameter(default=1)
    batch_size = luigi.IntParameter(default=50)
    input_shape = luigi.TupleParameter(default=(256, 256))
    val_batch_size = luigi.IntParameter(default=100)
    sampling_strategy = luigi.ChoiceParameter(choices=["oversample", "undersample"], default="oversample")
    interpolation = luigi.ChoiceParameter(choices=["nearest", "bilinear", "bicubic", "hamming", "box", "lanczos"],
                                          default="lanczos")
    keep_aspect_ratio = luigi.BoolParameter(default=False)
    keep_resized_for_training = luigi.BoolParameter(default=False)
    generator_workers = luigi.IntParameter(default=multiprocessing.cpu_count())
    generator_max_queue_size = luigi.IntParameter(default=20)
    data_augmented = luigi.BoolParameter(default=True)
    data_augmentation_params = luigi.DictParameter(default={})
    data_augmentation_width_resize_range = luigi.FloatParameter(default=0.0)
    trainable_head_layers = luigi.IntParameter(default=0)
    train_head = luigi.BoolParameter(default=False)

    def requires(self):
        return ResizeAndAugmentImages(val_size=self.val_size, sampling_strategy=self.sampling_strategy,
                                      target_size=self.input_shape,
                                      interpolation=self.interpolation, keep_aspect_ratio=self.keep_aspect_ratio,
                                      keep_resized_for_training=self.keep_resized_for_training,
                                      prior_augmentation=self.prior_augmentation,
                                      prior_augmentation_times=self.prior_augmentation_times)

    def output(self):
        return luigi.LocalTarget(_get_task_dir(self.__class__, self.task_id))

    def _prepare_dataset_if_necessary(self):
        if not os.path.exists(self.input().path):
            print(f"Gerando dataset {self.input().path}...")
            task_stack: List[luigi.Task] = []
            next_task: luigi.Task = self.requires()
            while next_task:
                if not next_task.complete():
                    task_stack.append(next_task)
                    next_task = next_task.requires()
                else:
                    break

            while task_stack:
                current_task = task_stack.pop()
                print(f"Rodando {current_task}...")
                current_task.run()
            print(f"Dataset {self.input().path} gerado com sucesso!")
        pass

    def get_train_generator(self):
        self._prepare_dataset_if_necessary()
        data_augmentation_params = self.data_augmentation_params.get("_FrozenOrderedDict__dict") \
            if "_FrozenOrderedDict__dict" in self.data_augmentation_params \
            else self.data_augmentation_params
        dir_name = "train" if not self.invert_train_val else "val"
        return create_train_generator(self.input().path, self.input_shape, self.interpolation, self.batch_size,
                                      dir_name, self.data_augmented, data_augmentation_params,
                                      self.data_augmentation_width_resize_range)

    def get_val_generator(self):
        self._prepare_dataset_if_necessary()
        dir_name = "val" if not self.invert_train_val else "train"
        return create_val_generator(self.input().path, self.input_shape, self.interpolation, self.val_batch_size,
                                    dir_name)

    def get_test_generator(self):
        self._prepare_dataset_if_necessary()
        return create_test_generator(self.input().path, self.input_shape, self.interpolation, self.val_batch_size)

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        try:
            with open(get_params_path(self.output().path), "w") as params_file:
                json.dump(self.param_kwargs, params_file, default=lambda o: o.__dict__, indent=4)

            train_generator = self.get_train_generator()
            val_generator = self.get_val_generator()

            training_result = self.fit(train_generator, val_generator)

            with open(get_training_result_path(self.output().path), "w") as training_log_file:
                training_log_file.write(training_result.to_json())

            plot_training_log(training_result.acc, training_result.val_acc, training_result.loss,
                              training_result.val_loss).savefig(get_training_log_path(self.output().path))
            plot_confusion_matrix(training_result.best_train_evaluation_report.confusion_matrix,
                                  ["NORMAL", "PNEUMONIA"]) \
                .savefig(get_train_confusion_matrix_path(self.output().path))
            plot_confusion_matrix(training_result.best_val_evaluation_report.confusion_matrix, ["NORMAL", "PNEUMONIA"]) \
                .savefig(get_val_confusion_matrix_path(self.output().path))
        except:
            shutil.rmtree(self.output().path)
            raise

    @abc.abstractmethod
    def fit(self, train_generator: DirectoryIterator, val_generator: DirectoryIterator) -> TrainingResult:
        """Realiza o treinamento e retorna uma instância de TrainingResult"""
        pass


class BaseKerasModel(BaseModel):
    __metaclass__ = abc.ABCMeta

    early_stopping_patience = luigi.IntParameter(default=20)
    epochs = luigi.IntParameter(default=100)
    number_of_trainings = luigi.IntParameter(default=1)

    def before_fit(self, model: Model, iteration: int = 0):
        """Permite alterar a rede antes de retreinar"""
        pass

    def fit(self, train_generator: DirectoryIterator, val_generator: DirectoryIterator) -> TrainingResult:
        try:
            self.keras_model = self.create_model()

            with open("%s/summary.txt" % self.output().path, "w") as summary_file:
                with redirect_stdout(summary_file):
                    self.keras_model.summary()

            model_weights = get_weights_path(self.output().path)
            checkpoint = ModelCheckpoint(model_weights, monitor='val_acc',
                                         verbose=1, save_best_only=True, mode='max')
            early_stopping = EarlyStopping(monitor='val_acc', patience=self.early_stopping_patience,
                                           verbose=1, mode='max')

            acc = []
            val_acc = []
            loss = []
            val_loss = []

            starting_time = time.time()

            for i in range(self.number_of_trainings):
                self.before_fit(self.keras_model, i)
                log = self.keras_model.fit_generator(train_generator, epochs=self.epochs,
                                                     steps_per_epoch=(
                                                         len(train_generator.filenames) / self.batch_size),
                                                     validation_data=val_generator,
                                                     validation_steps=(
                                                         len(val_generator.filenames) / self.val_batch_size),
                                                     verbose=1, callbacks=[checkpoint, early_stopping],
                                                     workers=self.generator_workers,
                                                     max_queue_size=self.generator_max_queue_size,
                                                     use_multiprocessing=True)
                acc += log.history["acc"]
                val_acc += log.history["val_acc"]
                loss += log.history["loss"]
                val_loss += log.history["val_loss"]

            ending_time = time.time()
            training_time = (ending_time - starting_time) / 60  # min

            self.keras_model.load_weights(model_weights)

            return TrainingResult(best_train_evaluation_report=evaluate_keras_model(self.keras_model, train_generator),
                                  best_val_evaluation_report=evaluate_keras_model(self.keras_model, val_generator),
                                  acc=acc, val_acc=val_acc, loss=loss, val_loss=val_loss, training_time=training_time)
        finally:
            K.clear_session()

    @abc.abstractmethod
    def create_model(self) -> Model:
        """Retornar um modelo do Keras compilado"""
        pass


class TransferLearningKerasModel(BaseKerasModel):
    __metaclass__ = abc.ABCMeta

    frozen_layers = luigi.IntParameter(default=9)
    retrain = luigi.BoolParameter(default=False)
    optimizer = luigi.Parameter(default="adam")
    optimizer_extra_params = luigi.DictParameter(default={})
    retrain_optimizer = luigi.Parameter(default="sgd")
    retrain_optimizer_extra_params = luigi.DictParameter(default={})
    learning_rate = luigi.FloatParameter(default=0.0001)
    retrain_learning_rate = luigi.FloatParameter(default=0.00001)

    @abc.abstractmethod
    def create_base_model(self) -> Model:
        """Retornar um modelo pré-treinado do Keras"""
        pass

    @abc.abstractmethod
    def create_model_with(self, base_model: Model) -> Model:
        """Retornar um modelo utilizando o modelo pré-treinado passado como parâmetro"""
        pass

    def create_model(self) -> Model:
        if self.retrain:
            self.number_of_trainings = 2

        self.base_model = self.create_base_model()
        
        if not self.train_head:
            for layer in self.base_model.layers[:self.frozen_layers]:
                layer.trainable = False
        else:
            for layer in self.base_model.layers[self.trainable_head_layers:]:
                layer.trainable = False

        self.keras_model = self.create_model_with(self.base_model)

        return self.keras_model

    def before_fit(self, model: Model, iteration: int = 0):
        if self.retrain and iteration == 1:
            optimizer = optimizers.get(
                {'class_name': self.retrain_optimizer,
                 'config': {**self.retrain_optimizer_extra_params, "lr": self.retrain_learning_rate}})
        else:
            optimizer = optimizers.get(
                {'class_name': self.optimizer,
                 'config': {**self.optimizer_extra_params, "lr": self.learning_rate}})

        if self.retrain:
            if iteration == 0:
                for layer in self.base_model.layers:
                    layer.trainable = False
            elif iteration == 1:
                print("Recarregando modelo de melhor val_acc...")
                weights_path = get_weights_path(self.output().path)
                model.load_weights(weights_path)
                for layer in self.base_model.layers[:self.frozen_layers]:
                    layer.trainable = False
                for layer in self.base_model.layers[self.frozen_layers:]:
                    layer.trainable = True
                # Salvando com as camadas congeladas
                model.save_weights(weights_path)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def _get_task_dir(model_cls: Type[BaseModel], task_id: str):
    return "output/models/%s/experiments/%s" % (model_cls.__name__, task_id)


def load_keras_model_from_task_dir(model_cls: Type[BaseKerasModel], task_dir: str) -> BaseKerasModel:
    model = model_cls(**get_params(task_dir))
    model.keras_model = model.create_model()
    model.keras_model.load_weights(get_weights_path(task_dir))
    return model


def load_keras_model(model_cls: Type[BaseKerasModel], task_id: str) -> BaseKerasModel:
    task_dir = _get_task_dir(model_cls, task_id)

    return load_keras_model_from_task_dir(model_cls, task_dir)
