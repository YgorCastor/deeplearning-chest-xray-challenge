from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

def plot_training_log(acc: List[float], val_acc: List[float], loss: List[float], val_loss: List[float]) -> Figure:
    fig = plt.figure()
    acc_ax = fig.add_subplot(2, 1, 1)
    acc_ax.plot(acc, label='acc')
    acc_ax.plot(val_acc, label='val_acc')
    acc_ax.legend()
    loss_ax = fig.add_subplot(2, 1, 2)
    loss_ax.plot(loss, label='loss')
    loss_ax.plot(val_loss, label='val_loss')
    loss_ax.legend()
    return fig


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names, figsize=(10, 7), fontsize=14) -> Figure:
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def plot_corrcoef_matrix(corrcoef_matrix: np.ndarray, model_names, figsize=(25, 15), fontsize=14) -> Figure:
    df_cm = pd.DataFrame(
        corrcoef_matrix, index=model_names, columns=model_names,
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt=".4f", cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=90, ha='center', fontsize=fontsize)
    return fig


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray, figsize=(10, 7)) -> Figure:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.step(recall, precision, color='b', alpha=0.2,
            where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2,
                    color='b')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall curve')
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, figsize=(10, 7)) -> Figure:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic curve')
    ax.legend(loc="lower right")
    return fig