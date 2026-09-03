"""
Heterogeneous Graph Transformer (HGT) for the HGNN Campaign Detection System.

Implements HGT with:
    1. Type-specific Q/K/V linear projections
    2. Multi-head attention with relation-type attention weights
    3. Residual connections + LayerNorm
    4. Per-type classifier heads

Reference: Hu et al., "Heterogeneous Graph Transformer" (WWW 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, List, Tuple, Optional


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer.

    Uses PyG's HGTConv which implements type-specific Q/K/V projections
    and multi-head attention with mutual attention mechanism.

    Architecture:
        Input → Type-specific Linear → [HGTConv + Residual + LayerNorm] × L → Classifier Heads
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        out_channels_dict: Optional[Dict[str, int]] = None,
        metadata: Optional[Tuple] = None,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels_dict: {node_type: input_feature_dim}
            hidden_channels: Hidden dimension
            out_channels_dict: {node_type: num_classes} for classification
            metadata: PyG metadata tuple (node_types, edge_types)
            num_layers: Number of HGTConv layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if out_channels_dict is None:
            out_channels_dict = {nt: 2 for nt in in_channels_dict}
        self.out_channels_dict = out_channels_dict

        # ── Input projections (type-specific) ──
        self.input_proj = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.input_proj[ntype] = Linear(in_dim, hidden_channels)

        # ── HGT Convolution layers ──
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
            )
            self.convs.append(conv)
            # Per-type layer normalization
            norm_dict = nn.ModuleDict()
            for ntype in in_channels_dict:
                norm_dict[ntype] = nn.LayerNorm(hidden_channels)
            self.norms.append(norm_dict)

        # ── Classifier heads (per node type) ──
        self.classifiers = nn.ModuleDict()
        for ntype, n_classes in out_channels_dict.items():
            self.classifiers[ntype] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, n_classes),
            )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_dict: {node_type: features [N_i, in_channels_i]}
            edge_index_dict: {(src, rel, dst): edge_index [2, E]}

        Returns:
            {node_type: logits [N_i, num_classes]} for classified node types
        """
        # ── Input projection ──
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_proj:
                h_dict[ntype] = F.relu(self.input_proj[ntype](x))
                h_dict[ntype] = F.dropout(h_dict[ntype], p=self.dropout,
                                          training=self.training)
            else:
                h_dict[ntype] = x

        # ── HGT layers with residual connections ──
        for i, conv in enumerate(self.convs):
            h_new = conv(h_dict, edge_index_dict)

            # Residual + LayerNorm
            for ntype in h_dict:
                if ntype in h_new and ntype in self.norms[i]:
                    h_new[ntype] = self.norms[i][ntype](
                        h_new[ntype] + h_dict[ntype]  # residual
                    )
                    h_new[ntype] = F.dropout(h_new[ntype], p=self.dropout,
                                            training=self.training)

            h_dict = h_new

        # ── Classification ──
        out_dict = {}
        for ntype in self.out_channels_dict:
            if ntype in h_dict and ntype in self.classifiers:
                out_dict[ntype] = self.classifiers[ntype](h_dict[ntype])

        return out_dict

    def get_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Get node embeddings without classification (for downstream tasks).

        Returns:
            {node_type: embeddings [N_i, hidden_channels]}
        """
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_proj:
                h_dict[ntype] = F.relu(self.input_proj[ntype](x))
            else:
                h_dict[ntype] = x

        for i, conv in enumerate(self.convs):
            h_new = conv(h_dict, edge_index_dict)
            for ntype in h_dict:
                if ntype in h_new and ntype in self.norms[i]:
                    h_new[ntype] = self.norms[i][ntype](
                        h_new[ntype] + h_dict[ntype]
                    )
            h_dict = h_new

        return h_dict
