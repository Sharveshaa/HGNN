"""
Hyperparameter Tuning for the HGNN Campaign Detection System.
"""

import itertools
import os
import json
import torch
from .trainer import HGNNTrainer
import copy

class HyperparameterTuner:
    """
    Grid search hyperparameter tuner for HGNN models.
    """
    
    def __init__(self, model_class, base_model_kwargs, trainer_kwargs, search_space):
        self.model_class = model_class
        self.base_model_kwargs = base_model_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.search_space = search_space
        self.results = []
        
    def _generate_configs(self):
        keys = self.search_space.keys()
        values = self.search_space.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
            
    def tune(self, data, save_dir="tuning_results"):
        os.makedirs(save_dir, exist_ok=True)
        
        configs = list(self._generate_configs())
        print(f"Starting grid search over {len(configs)} configurations...")
        
        best_f1 = -1
        best_config = None
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Testing config: {config}")
            
            # Prepare model args
            model_kwargs = copy.deepcopy(self.base_model_kwargs)
            
            # Update model_kwargs with config if applicable
            for k, v in config.items():
                if k in model_kwargs:
                    model_kwargs[k] = v
                    
            model = self.model_class(**model_kwargs)
            
            # Prepare trainer args
            trainer_args = copy.deepcopy(self.trainer_kwargs)
            trainer_args['model'] = model
            if 'lr' in config:
                trainer_args['lr'] = config['lr']
            if 'weight_decay' in config:
                trainer_args['weight_decay'] = config['weight_decay']
                
            trainer = HGNNTrainer(**trainer_args)
            
            # Train (reduced epochs/patience for tuning)
            try:
                test_metrics = trainer.train(
                    data, 
                    epochs=config.get('epochs', 50), 
                    patience=10, 
                    log_every=10,
                    save_dir=os.path.join(save_dir, f"run_{i}")
                )
                
                # Compute average macro F1 across targets
                avg_f1 = 0
                count = 0
                for nt, m in test_metrics.items():
                    if 'f1_macro' in m:
                        avg_f1 += m['f1_macro']
                        count += 1
                
                if count > 0:
                    avg_f1 /= count
                    
                self.results.append({
                    'config': config,
                    'avg_f1': avg_f1,
                    'metrics': test_metrics
                })
                
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_config = config
                    print(f"  >>> New best! Avg Macro-F1: {best_f1:.4f}")
                    
            except Exception as e:
                print(f"Error training config {config}: {e}")
                
        # Save results
        with open(os.path.join(save_dir, 'tuning_summary.json'), 'w') as f:
            def default(obj):
                import numpy as np
                if type(obj).__module__ == np.__name__:
                    return obj.item() if np.isscalar(obj) else obj.tolist()
                raise TypeError
            json.dump({'best_config': best_config, 'best_f1': best_f1, 'all_results': self.results}, f, default=default, indent=2)
            
        print(f"\nTuning complete! Best Avg Macro-F1: {best_f1:.4f}")
        print(f"Best config: {best_config}")
        
        return best_config, self.results
