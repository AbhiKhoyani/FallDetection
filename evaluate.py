import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, X, y, labels = [0,1]):

    metrics = [accuracy_score, precision_score, recall_score, f1_score, None]
    metric_name = ['accuracy', 'precision', 'recall', 'f1', 'specificity']

    y_prob = model.predict(X)
    y_pred = np.where(y_prob < 0.5, 0, 1)
    y_prob = np.vstack([1 - y_prob[:,0], y_prob[:,0]]).T
    assert y_pred.shape[1] == 1, "Model predication shape shall be (batch_size, 1)."
    for i,j in zip(metric_name, metrics):
        if i == 'specificity':
            wandb.summary[i] = recall_score(y, y_pred, pos_label=0)
        else:
            wandb.summary[i] = j(y, y_pred)    

    wandb.sklearn.plot_confusion_matrix(y, y_pred, labels)
    wandb.sklearn.plot_roc(y, y_prob, labels)
    wandb.sklearn.plot_precision_recall(y, y_prob, labels)