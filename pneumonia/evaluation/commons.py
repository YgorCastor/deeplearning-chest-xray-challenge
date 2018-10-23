from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
import json
import ntpath

from pneumonia.files import get_params, get_training_result_path, listdir_fullpath, get_params_path


class EvaluationReport(object):
    def __init__(self, acc: float, precision: float, recall: float, f1_score: float,
                 confusion_matrix: List[List[float]], misclassified_inputs: Dict[str, float] = None):
        self.acc = acc
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.confusion_matrix = confusion_matrix
        self.misclassified_inputs = misclassified_inputs


class TrainingResult(object):
    def __init__(self, best_train_evaluation_report: EvaluationReport, best_val_evaluation_report: EvaluationReport,
                 acc: List[float], val_acc: List[float], loss: List[float],
                 val_loss: List[float], training_time: float):
        self.best_train_evaluation_report = best_train_evaluation_report
        self.best_val_evaluation_report = best_val_evaluation_report
        self.acc = acc
        self.val_acc = val_acc
        self.loss = loss
        self.val_loss = val_loss
        self.training_time = training_time

    @classmethod
    def from_json(cls, json_obj: dict) -> 'TrainingResult':
        training_result = TrainingResult(**json_obj)
        training_result.best_train_evaluation_report = EvaluationReport(**training_result.best_train_evaluation_report)
        training_result.best_val_evaluation_report = EvaluationReport(**training_result.best_val_evaluation_report)
        return training_result

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)


def _get_summary_of_experiment(task_dir: str) -> dict:
    experiment_summary = {"name": ntpath.basename(task_dir)}
    params = get_params(task_dir)
    with open(get_training_result_path(task_dir), 'r') as f:
        training_result = TrainingResult.from_json(json.load(f))
    experiment_summary = {**experiment_summary, **params}
    experiment_summary["train_acc"] = training_result.best_train_evaluation_report.acc
    experiment_summary["train_precision"] = training_result.best_train_evaluation_report.precision
    experiment_summary["train_recall"] = training_result.best_train_evaluation_report.recall
    experiment_summary["train_f1_score"] = training_result.best_train_evaluation_report.f1_score
    experiment_summary["val_acc"] = training_result.best_val_evaluation_report.acc
    experiment_summary["val_precision"] = training_result.best_val_evaluation_report.precision
    experiment_summary["val_recall"] = training_result.best_val_evaluation_report.recall
    experiment_summary["val_f1_score"] = training_result.best_val_evaluation_report.f1_score
    experiment_summary["training_time"] = training_result.training_time

    return experiment_summary


def calculate_scores(trues: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float, float, List[List[float]]]:
    acc = metrics.accuracy_score(trues, preds)
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(trues, preds, average='binary')
    confusion_matrix = metrics.confusion_matrix(trues, preds).tolist()

    return acc, precision, recall, f1_score, confusion_matrix

def generate_summary_of_experiments(task_dirs: List[str]) -> pd.DataFrame:
    return pd.DataFrame([_get_summary_of_experiment(task_dir) for task_dir in task_dirs])


def generate_summary_of_experiments_from_dir(directory: str) -> pd.DataFrame:
    return generate_summary_of_experiments(list_experiment_dirs(directory))


def list_experiment_dirs(directory: str):
    return [directory for directory in listdir_fullpath(directory) if ntpath.exists(get_training_result_path(directory))]