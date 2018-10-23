import os
import json


def get_params_path(task_dir: str) -> str:
    return "%s/params.json" % task_dir


def get_weights_path(task_dir: str) -> str:
    return "%s/model.h5" % task_dir


def get_training_log_path(task_dir: str) -> str:
    return "%s/training_log.jpg" % task_dir


def get_training_result_path(task_dir: str) -> str:
    return "%s/training_result.json" % task_dir


def get_val_confusion_matrix_path(task_dir: str) -> str:
    return "%s/val_confusion_matrix.jpg" % task_dir


def get_train_confusion_matrix_path(task_dir: str) -> str:
    return "%s/train_confusion_matrix.jpg" % task_dir


def get_params(task_dir: str) -> dict:
    with open(get_params_path(task_dir), "r") as params_file:
        return json.load(params_file)

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]