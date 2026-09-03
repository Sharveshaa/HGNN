"""
Unified Trainer for the HGNN Campaign Detection System.
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List, Any, Optional
import json

from .metrics import compute_metrics, MetricTracker

class HGNNTrainer:
    """
    Unified trainer for all HGNN model variants (GraphSAGE, GAT, HAN, HGT).
    Supports multi-task learning (classifying multiple node types simultaneously).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_node_types: List[str] = ['user', 'post', 'url', 'campaign'],
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        optimizer_type: str = 'adam',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.target_node_types = target_node_types
        
        # Optimizer setup
        if optimizer_type.lower() == 'adamw':
            self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
        self.class_weights = class_weights or {}
        for nt, w in self.class_weights.items():
            self.class_weights[nt] = w.to(self.device)
            
        self.task_weights = task_weights or {nt: 1.0 for nt in target_node_types}
        self.tracker = MetricTracker()
        
    def _create_masks(self, data, train_ratio=0.7, val_ratio=0.15):
        """Create standard train/val/test splits."""
        for nt in self.target_node_types:
            if nt not in data.node_types or not hasattr(data[nt], 'y'):
                continue
                
            num_nodes = data[nt].y.size(0)
            if num_nodes == 0:
                continue
                
            indices = torch.randperm(num_nodes)
            train_end = int(train_ratio * num_nodes)
            val_end = train_end + int(val_ratio * num_nodes)
            
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]
            
            data[nt].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data[nt].train_mask[train_idx] = True
            
            data[nt].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data[nt].val_mask[val_idx] = True
            
            data[nt].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data[nt].test_mask[test_idx] = True

    def _compute_loss(self, out_dict, data, mask_name):
        """Compute weighted multi-task loss."""
        total_loss = 0
        loss_dict = {}
        
        for nt in self.target_node_types:
            if nt not in out_dict or nt not in data.node_types or not hasattr(data[nt], 'y'):
                continue
                
            if not hasattr(data[nt], mask_name):
                continue
                
            mask = data[nt][mask_name]
            if mask.sum() == 0:
                continue
                
            pred = out_dict[nt][mask]
            target = data[nt].y[mask]
            
            # Apply class weights if available
            weight = self.class_weights.get(nt, None)
            
            loss = F.cross_entropy(pred, target, weight=weight)
            weighted_loss = loss * self.task_weights.get(nt, 1.0)
            
            total_loss += weighted_loss
            loss_dict[nt] = loss.item()
            
        return total_loss, loss_dict
        
    def _evaluate(self, out_dict, data, mask_name):
        """Evaluate metrics for a given split."""
        metrics_dict = {}
        
        for nt in self.target_node_types:
            if nt not in out_dict or not hasattr(data[nt], mask_name):
                continue
                
            mask = data[nt][mask_name]
            if mask.sum() == 0:
                continue
                
            logits = out_dict[nt][mask]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            targets = data[nt].y[mask].cpu().numpy()
            
            num_classes = logits.size(1)
            metrics_dict[nt] = compute_metrics(targets, preds, probs, num_classes)
            
        return metrics_dict
        
    def train(self, data, epochs=100, patience=20, log_every=10, save_dir="models/checkpoints"):
        """Train the model with early stopping."""
        print(f"Training on device: {self.device}")
        
        # Ensure data is on device
        data = data.to(self.device)
        
        # Create splits if they don't exist
        if not hasattr(data['user'], 'train_mask'):
            self._create_masks(data)
            
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'best_model.pth')
        
        for epoch in range(1, epochs + 1):
            # ── Train ──
            self.model.train()
            self.optimizer.zero_grad()
            
            out = self.model(data.x_dict, data.edge_index_dict)
            loss, train_loss_dict = self._compute_loss(out, data, 'train_mask')
            
            if type(loss) == int and loss == 0:
                print("Warning: No valid training targets found.")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # ── Eval ──
            self.model.eval()
            with torch.no_grad():
                out_eval = self.model(data.x_dict, data.edge_index_dict)
                val_loss, val_loss_dict = self._compute_loss(out_eval, data, 'val_mask')
                
                train_metrics = self._evaluate(out_eval, data, 'train_mask')
                val_metrics = self._evaluate(out_eval, data, 'val_mask')
                
            is_best = self.tracker.update(epoch, train_metrics, val_metrics)
            
            if is_best:
                torch.save(self.model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % log_every == 0 or epoch == 1:
                val_f1s = [f"{nt}: {m.get('f1_macro', 0):.4f}" for nt, m in val_metrics.items()]
                print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Macro-F1: {', '.join(val_f1s)}")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            scheduler.step()
            
        # Load best model for testing
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x_dict, data.edge_index_dict)
            test_metrics = self._evaluate(out, data, 'test_mask')
            
        print("\n=== Test Results ===")
        for nt, m in test_metrics.items():
            print(f"\n{nt.upper()}:")
            print(f"  Accuracy: {m.get('accuracy', 0):.4f}")
            print(f"  Macro-F1: {m.get('f1_macro', 0):.4f}")
            print(f"  AUC:      {m.get('auc', 0):.4f}")
            
        # Save metrics
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            # handle numpy types
            def default(obj):
                if type(obj).__module__ == np.__name__:
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    else:
                        return obj.item()
                raise TypeError('Unknown type:', type(obj))
            json.dump({'test': test_metrics, 'history': self.tracker.history}, f, default=default, indent=2)
            
        return test_metrics
