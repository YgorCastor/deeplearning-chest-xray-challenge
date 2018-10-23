from keras.engine.training import Model
from keras.preprocessing.image import DirectoryIterator
import numpy as np
import contextlib

from pneumonia.evaluation.commons import EvaluationReport, calculate_scores


def pred_probas(model: Model, generator: DirectoryIterator) -> np.ndarray:
    with _ordered(generator) as ordered_generator:
        steps = len(ordered_generator.filenames) / generator.batch_size
        return model.predict_generator(ordered_generator, steps=steps, verbose=1).flatten()


def evaluate_keras_model(model: Model, generator: DirectoryIterator, threshold=0.5) -> EvaluationReport:
    with _ordered(generator) as ordered_generator:
        trues = ordered_generator.classes
        proba_preds = pred_probas(model, ordered_generator)
        preds = (proba_preds > threshold).astype(int)

        acc, precision, recall, f1_score, confusion_matrix = calculate_scores(trues, preds)

        misclassified_filenames = np.array(ordered_generator.filenames)[trues != preds]
        misclassified_proba_preds = proba_preds[trues != preds]
        misclassified_inputs = {filename: float(proba_pred) for filename, proba_pred in
                                zip(misclassified_filenames, misclassified_proba_preds)}

        return EvaluationReport(acc, precision, recall, f1_score, confusion_matrix, misclassified_inputs)


@contextlib.contextmanager
def _ordered(generator: DirectoryIterator) -> DirectoryIterator:
    old_shuffle = generator.shuffle

    generator.shuffle = False
    generator.on_epoch_end()

    yield generator

    generator.shuffle = old_shuffle
    generator.on_epoch_end()