import os
from typing import Tuple, List

import numpy as np
from pneumonia.model.vgg16 import VGG16
from pneumonia.model.vgg19 import VGG19
from pneumonia.model.densenet121 import DENSENET121
from pneumonia.model.base import load_keras_model_from_task_dir
from pneumonia.evaluation.keras import pred_probas
from pneumonia.evaluation.commons import list_experiment_dirs

from keras import backend as K

PATH_TO_MODEL = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "DENSENET121": DENSENET121,
}


class ModelPredictions(object):
    def __init__(self, name: str, val_probas: np.ndarray, test_probas: np.ndarray,
                 val_preds: np.ndarray, test_preds: np.ndarray) -> None:
        self.name = name
        self.val_probas = val_probas
        self.test_probas = test_probas
        self.val_preds = val_preds
        self.test_preds = test_preds


def get_trained_models_predictions_and_trues() -> Tuple[List[ModelPredictions], np.ndarray, np.ndarray]:
    val_trues: np.ndarray = None
    test_trues: np.ndarray = None

    model_names = [os.path.basename(experiment_dir) for experiment_dir in list_experiment_dirs("trained_models")]
    model_predictions: List[ModelPredictions] = []
    for model_name in model_names:
        task_dir = os.path.join("trained_models", model_name)
        model_cls = [cls for name, cls in PATH_TO_MODEL.items() if model_name.startswith(name)][0]
        model = load_keras_model_from_task_dir(model_cls, task_dir)

        if val_trues is None:
            val_trues = model.get_val_generator().classes
            test_trues = model.get_test_generator().classes

        val_probas = pred_probas(model.keras_model, model.get_val_generator())
        test_probas = pred_probas(model.keras_model, model.get_test_generator())
        K.clear_session()

        threshold_path = os.path.join(task_dir, "threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as threshold_file:
                threshold = float(threshold_file.read())
            val_preds = (val_probas > threshold).astype(int)
            test_preds = (test_probas > threshold).astype(int)

            model_predictions.append(ModelPredictions(model_name, val_probas=val_probas, test_probas=test_probas,
                                                      val_preds=val_preds, test_preds=test_preds))
        else:
            print(f"Arquivo threshold.txt não encontrado para {model_name}")

    return model_predictions, val_trues, test_trues
