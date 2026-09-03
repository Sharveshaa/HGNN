"""
Experiment runner for Temporal HGNN.
"""

import os
import sys
import torch
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph.construct_graph import build_hetero_graph
from graph.temporal import TemporalHeteroDataset, TemporalConfig
from models.temporal_hgnn import TemporalHGNN
from training.trainer import HGNNTrainer
import warnings
warnings.filterwarnings("ignore")

def run_temporal(data_dir="data"):
    print("Loading Base Graph...")
    data, mappings = build_hetero_graph(
        raw_dir=os.path.join(data_dir, "raw"),
        processed_dir=os.path.join(data_dir, "processed"),
        use_rich_features=True
    )
    
    print("Building Temporal Snapshots...")
    posts_df = pd.read_csv(os.path.join(data_dir, "raw", "posts.csv"))
    users_df = pd.read_csv(os.path.join(data_dir, "raw", "users.csv"))
    
    config = TemporalConfig(snapshot_hours=8760.0, min_nodes_per_snapshot=5)
    temporal_dataset = TemporalHeteroDataset(config)
    
    snapshots = temporal_dataset.build_snapshots(
        data, posts_df, users_df, mappings['post'], mappings['user']
    )
    
    print(f"Created {len(snapshots)} snapshots.")
    if len(snapshots) < 2:
        print("Not enough snapshots for temporal modeling. Exiting.")
        return
        
    in_channels_dict = {nt: data[nt].x.size(1) for nt in data.node_types if hasattr(data[nt], 'x')}
    out_channels_dict = {
        'user': 2,
        'post': 2,
        'campaign': 3,
    }
    
    model = TemporalHGNN(
        in_channels_dict=in_channels_dict,
        hidden_channels=128,
        out_channels_dict=out_channels_dict,
        metadata=data.metadata(),
        num_gnn_layers=2,
        num_gru_layers=1,
        bidirectional=False
    )
    
    # We need a custom training loop or adapter for TemporalHGNN since it takes a list of snapshots
    # For simplicity in this runner, we'll create an adapter that looks like a normal model
    # but feeds the temporal sequence internally.
    
    class TemporalAdapter(torch.nn.Module):
        def __init__(self, temp_model, snapshots_list):
            super().__init__()
            self.model = temp_model
            self.snapshots = snapshots_list
            
        def forward(self, x_dict, edge_index_dict):
            # Ignore inputs, use the stored sequence
            # x_dict/edge_index_dict are from the 'aggregated' data object passed by trainer
            return self.model(self.snapshots)
            
    adapter = TemporalAdapter(model, snapshots)
    
    # We train against the final state (the aggregated graph `data`)
    print(f"\n{'='*50}\nTraining Temporal HGNN\n{'='*50}")
    
    trainer = HGNNTrainer(
        model=adapter,
        target_node_types=list(out_channels_dict.keys()),
        lr=0.001,
        weight_decay=1e-4,
    )
    
    # The trainer uses 'data' to get labels and masks.
    # The adapter ignores data.x_dict and uses the snapshots list.
    metrics = trainer.train(
        data,
        epochs=50,
        patience=10,
        log_every=5,
        save_dir=os.path.join("models", "experiments", "temporal")
    )
    
    print("Temporal experiment complete.")

if __name__ == "__main__":
    run_temporal()
