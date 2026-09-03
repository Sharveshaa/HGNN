"""
Temporal HGNN for the HGNN Campaign Detection System.

Processes sequences of graph snapshots using:
    1. Per-snapshot: HGT/HAN encoder → node embeddings
    2. Cross-snapshot: GRU over temporal node embedding sequences
    3. Final: Temporal-aware node classification + campaign trajectory prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional

from models.hgt import HGT


class TemporalGRUEncoder(nn.Module):
    """
    Encodes temporal sequences of node embeddings using a GRU.

    For each node, processes its embedding sequence across T snapshots
    to produce a temporally-aware representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [N_nodes, T_snapshots, input_dim]

        Returns:
            output: [N_nodes, output_dim] (last hidden state)
        """
        # GRU output: [N, T, hidden*directions]
        output, _ = self.gru(sequence)
        # Use last timestep
        return output[:, -1, :]


class TemporalHGNN(nn.Module):
    """
    Temporal Heterogeneous Graph Neural Network.

    Architecture:
        For each snapshot t:
            1. HGT encoder: x_dict → embeddings_dict
        Across snapshots:
            2. Stack temporal embedding sequences per node
            3. GRU encoder: [emb_t1, emb_t2, ...] → temporal embedding
        Final:
            4. Classifier heads for each node type

    This model captures both structural (intra-snapshot) and temporal
    (cross-snapshot) patterns in the heterogeneous graph.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        out_channels_dict: Optional[Dict[str, int]] = None,
        metadata: Optional[Tuple] = None,
        num_gnn_layers: int = 2,
        num_gru_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Args:
            in_channels_dict: {node_type: input_feature_dim}
            hidden_channels: Hidden dimension for both GNN and GRU
            out_channels_dict: {node_type: num_classes}
            metadata: PyG metadata for HGT
            num_gnn_layers: Layers in the per-snapshot GNN encoder
            num_gru_layers: Layers in the temporal GRU
            num_heads: Attention heads for HGT
            dropout: Dropout probability
            bidirectional: Whether GRU is bidirectional
        """
        super().__init__()
        self.hidden_channels = hidden_channels

        if out_channels_dict is None:
            out_channels_dict = {nt: 2 for nt in in_channels_dict}
        self.out_channels_dict = out_channels_dict

        # ── Per-snapshot GNN encoder ──
        # Returns embeddings, not logits (we classify after temporal encoding)
        self.gnn_encoder = HGT(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            out_channels_dict=None,  # No classifier; we get embeddings
            metadata=metadata,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Remove the classifier from GNN — we use get_embeddings instead

        # ── Temporal GRU encoders (per node type) ──
        self.temporal_encoders = nn.ModuleDict()
        gru_output_dim = hidden_channels * (2 if bidirectional else 1)

        for ntype in in_channels_dict:
            self.temporal_encoders[ntype] = TemporalGRUEncoder(
                input_dim=hidden_channels,
                hidden_dim=hidden_channels,
                num_layers=num_gru_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )

        # ── Classifier heads ──
        self.classifiers = nn.ModuleDict()
        for ntype, n_classes in out_channels_dict.items():
            self.classifiers[ntype] = nn.Sequential(
                nn.Linear(gru_output_dim, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, n_classes),
            )

    def encode_snapshot(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Encode a single snapshot using HGT."""
        return self.gnn_encoder.get_embeddings(x_dict, edge_index_dict)

    def forward(
        self,
        snapshots: List[HeteroData],
        target_node_types: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass over a sequence of temporal snapshots.

        Args:
            snapshots: List of HeteroData objects (temporal sequence)
            target_node_types: Which node types to classify

        Returns:
            {node_type: logits} for the LAST snapshot's nodes
        """
        if target_node_types is None:
            target_node_types = list(self.out_channels_dict.keys())

        if len(snapshots) == 0:
            return {}

        # ── Step 1: Encode each snapshot ──
        snapshot_embeddings = []
        for snap in snapshots:
            emb = self.encode_snapshot(snap.x_dict, snap.edge_index_dict)
            snapshot_embeddings.append(emb)

        # ── Step 2: Build temporal sequences per node type ──
        # For persistent nodes (same count across snapshots), we can
        # directly stack. For varying-size nodes, we use the last snapshot's
        # node count and pad earlier snapshots.

        out_dict = {}

        for ntype in target_node_types:
            if ntype not in self.temporal_encoders:
                continue

            # Get node counts per snapshot
            node_counts = []
            for emb_dict in snapshot_embeddings:
                if ntype in emb_dict:
                    node_counts.append(emb_dict[ntype].shape[0])
                else:
                    node_counts.append(0)

            if all(c == 0 for c in node_counts):
                continue

            # Use the last snapshot's node count as reference
            last_count = node_counts[-1]
            if last_count == 0:
                continue

            T = len(snapshots)
            device = snapshot_embeddings[-1][ntype].device

            # Build temporal sequence [N_nodes, T, hidden_channels]
            temporal_seq = torch.zeros(
                last_count, T, self.hidden_channels, device=device
            )

            for t, emb_dict in enumerate(snapshot_embeddings):
                if ntype in emb_dict:
                    emb = emb_dict[ntype]
                    n = min(emb.shape[0], last_count)
                    temporal_seq[:n, t, :] = emb[:n]

            # ── Step 3: GRU encoding ──
            temporal_emb = self.temporal_encoders[ntype](temporal_seq)

            # ── Step 4: Classification ──
            if ntype in self.classifiers:
                out_dict[ntype] = self.classifiers[ntype](temporal_emb)

        return out_dict

    def forward_single(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass on a single snapshot (non-temporal fallback).

        Useful for inference when only the current graph state is available.
        """
        emb_dict = self.encode_snapshot(x_dict, edge_index_dict)

        out_dict = {}
        for ntype in self.out_channels_dict:
            if ntype in emb_dict and ntype in self.classifiers:
                # Pass through GRU with sequence length 1
                emb = emb_dict[ntype].unsqueeze(1)  # [N, 1, H]
                temporal_emb = self.temporal_encoders[ntype](emb)
                out_dict[ntype] = self.classifiers[ntype](temporal_emb)

        return out_dict
