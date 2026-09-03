"""
Experiment runner for comparing baselines (GraphSAGE, GAT, HAN, HGT).
"""

import os
import sys
import torch
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph.construct_graph import build_hetero_graph
from models.baselines import PhishingHGNN, GATBaseline
from models.han import HAN
from models.hgt import HGT
from training.trainer import HGNNTrainer
import warnings
warnings.filterwarnings("ignore")

def run_baselines(data_dir="data"):
    print("Loading Graph...")
    data, mappings = build_hetero_graph(
        raw_dir=os.path.join(data_dir, "raw"),
        processed_dir=os.path.join(data_dir, "processed"),
        use_rich_features=True
    )
    
    in_channels_dict = {nt: data[nt].x.size(1) for nt in data.node_types if hasattr(data[nt], 'x')}
    out_channels_dict = {
        'user': 2,
        'post': 2,
        'url': 2,
        'campaign': 3,
    }
    
    # Calculate class weights for imbalance
    class_weights = {}
    for nt, n_classes in out_channels_dict.items():
        if hasattr(data[nt], 'y'):
            y = data[nt].y
            counts = torch.bincount(y, minlength=n_classes)
            total = counts.sum().float()
            # Inverse frequency weighting
            weights = total / (n_classes * counts.float() + 1e-6)
            class_weights[nt] = weights

    models_to_run = {
        'GraphSAGE': PhishingHGNN(
            hidden_channels=128, 
            out_channels=2, # not fully used with out_channels_dict below, handled inside wrapper if needed, but PhishingHGNN is simple
            metadata=data.metadata(),
            num_layers=2
        ),
        'GAT': GATBaseline(
            hidden_channels=128,
            out_channels=2,
            metadata=data.metadata(),
            num_layers=2,
            heads=4
        ),
        'HAN': HAN(
            in_channels_dict=in_channels_dict,
            hidden_channels=128,
            out_channels_dict=out_channels_dict,
            num_layers=2,
            heads=4
        ),
        'HGT': HGT(
            in_channels_dict=in_channels_dict,
            hidden_channels=128,
            out_channels_dict=out_channels_dict,
            metadata=data.metadata(),
            num_layers=3,
            num_heads=4
        )
    }
    
    # Adapt simple baselines to multi-task output
    class MultiTaskWrapper(torch.nn.Module):
        def __init__(self, base_model, out_dict, hidden_dim):
            super().__init__()
            self.base = base_model
            self.classifiers = torch.nn.ModuleDict({
                nt: torch.nn.Linear(hidden_dim, num_c)
                for nt, num_c in out_dict.items()
            })
            
        def forward(self, x_dict, edge_index_dict):
            # Hack for PhishingHGNN/GATBaseline since they output fixed dims
            # Assuming they were initialized with out_channels=hidden_channels
            h_dict = self.base(x_dict, edge_index_dict)
            out = {}
            for nt, clf in self.classifiers.items():
                if nt in h_dict:
                    out[nt] = clf(h_dict[nt])
            return out
            
    # Wrap GraphSAGE and GAT
    models_to_run['GraphSAGE'] = MultiTaskWrapper(
        PhishingHGNN(128, 128, data.metadata(), 2), 
        out_channels_dict, 128
    )
    models_to_run['GAT'] = MultiTaskWrapper(
        GATBaseline(128, 128, data.metadata(), 2, 4), 
        out_channels_dict, 128
    )

    results = {}
    
    for name, model in models_to_run.items():
        print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
        
        trainer = HGNNTrainer(
            model=model,
            target_node_types=list(out_channels_dict.keys()),
            lr=0.001,
            weight_decay=1e-4,
            class_weights=class_weights,
            task_weights={'campaign': 2.0, 'user': 1.0, 'post': 1.0, 'url': 1.0} # Prioritize campaign
        )
        
        metrics = trainer.train(
            data,
            epochs=100,
            patience=20,
            log_every=10,
            save_dir=os.path.join("models", "experiments", name)
        )
        
        results[name] = metrics
        
    # Summarize
    print("\n\n=== Final Comparison ===")
    summary = []
    for name, res in results.items():
        row = {'Model': name}
        for nt in out_channels_dict.keys():
            if nt in res and 'f1_macro' in res[nt]:
                row[f'{nt}_F1'] = res[nt]['f1_macro']
        summary.append(row)
        
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    
    os.makedirs(os.path.join("models", "experiments"), exist_ok=True)
    df.to_csv(os.path.join("models", "experiments", "baseline_comparison.csv"), index=False)

if __name__ == "__main__":
    run_baselines()
