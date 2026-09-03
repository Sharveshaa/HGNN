"""
Evaluation Metrics for the HGNN Campaign Detection System.
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from typing import Dict, Any


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None, num_classes: int = 2) -> Dict[str, Any]:
    """
    Compute classification metrics.
    """
    if len(y_true) == 0:
        return {}
        
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = (y_true == y_pred).mean()
    
    # Precision, Recall, F1 (macro and weighted)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics['f1_macro'] = f1_macro
    metrics['precision_macro'] = p_macro
    metrics['recall_macro'] = r_macro
    
    metrics['f1_weighted'] = f1_wt
    metrics['precision_weighted'] = p_wt
    metrics['recall_weighted'] = r_wt
    
    # Per-class metrics
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    for c in range(num_classes):
        metrics[f'class_{c}_f1'] = f1[c]
        metrics[f'class_{c}_support'] = support[c]
        
    # AUC if probabilities provided
    if y_prob is not None:
        try:
            if num_classes == 2:
                # Binary AUC expects probabilities of positive class
                if y_prob.shape[1] == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class AUC
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except Exception:
            metrics['auc'] = 0.0
            
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    
    return metrics


class MetricTracker:
    """Tracks metrics across epochs."""
    def __init__(self):
        self.history = {}
        self.best_epoch = -1
        self.best_val_f1 = -1.0
        
    def update(self, epoch: int, train_metrics: Dict[str, Any], val_metrics: Dict[str, Any]):
        self.history[epoch] = {
            'train': train_metrics,
            'val': val_metrics
        }
        
        # Track overall best macro F1 across all target types
        val_f1 = np.mean([
            m['f1_macro'] for m in val_metrics.values() if 'f1_macro' in m
        ])
        
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_epoch = epoch
            return True # is best
        return False
        
    def get_best_metrics(self):
        if self.best_epoch == -1:
            return None
        return self.history[self.best_epoch]
