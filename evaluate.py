import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X, y, labels = [0,1]):

    metrics = [accuracy_score, precision_score, recall_score, f1_score]
    metric_name = ['Accuracy', 'Precision', 'Recall', 'F1']

    y_pred = model.predict(X)
    assert y_pred.shape[1] == 2, "Model predication shape shall be (batch_size, 2)."
    for i,j in zip(metric_name, metrics):
        wandb.summary[i] = j(y, np.argmin(y_pred, axis = -1))    

    wandb.sklearn.plot_confusion_matrix(y, np.argmin(y_pred, axis = -1), labels)
    wandb.sklearn.plot_roc(y, y_pred, labels)
    wandb.sklearn.plot_precision_recall(y, y_pred, labels)