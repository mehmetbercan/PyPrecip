import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix
import math

def calc_metrics(y_true_cls, y_pred_cls, class_means):
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=range(len(class_means)))
    total = cm.sum()
    hits = np.trace(cm)
    falsealrm = np.triu(cm, k=1).sum()
    misses = np.tril(cm, k=-1).sum()

    acc = np.trace(cm) / total if total > 0 else float('nan')
    pod = hits / (hits + misses) if (hits + misses) > 0 else float('nan')
    far = falsealrm / (falsealrm + hits) if (falsealrm + hits) > 0 else float('nan')
    csi = hits / (hits + falsealrm + misses) if (hits + falsealrm + misses) > 0 else float('nan')

    y_true_mm = np.array([class_means[c] for c in y_true_cls])
    y_pred_mm = np.array([class_means[c] for c in y_pred_cls])
    rmse = math.sqrt(mean_squared_error(y_true_mm, y_pred_mm))
    rse = rmse / y_true_mm.std(ddof=0) if y_true_mm.std(ddof=0) > 0 else float('nan')

    return {"ACC": acc, "RMSE": rmse, "RSE": rse, "POD": pod, "FAR": far, "CSI": csi, "ConfusionMatrix": cm}