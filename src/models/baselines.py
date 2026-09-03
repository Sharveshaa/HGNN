"""
Baseline Models for the HGNN Campaign Detection System.

Contains the original GraphSAGE-based HGNN baseline and a GAT baseline,
both converted to heterogeneous via PyG's to_hetero.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, to_hetero


class BaseGNN(torch.nn.Module):
    """Base homogeneous GraphSAGE model."""

    def __init__(self, hidden_channels, out_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
        self.convs.append(SAGEConv((-1, -1), out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class PhishingHGNN(torch.nn.Module):
    """
    Heterogeneous GraphSAGE baseline.

    Wraps a homogeneous GNN via to_hetero() to create separate weight
    matrices for each edge type, with mean aggregation across types.
    """

    def __init__(self, hidden_channels, out_channels, metadata,
                 num_layers=2, dropout=0.3, aggr='mean'):
        super().__init__()
        base = BaseGNN(hidden_channels, out_channels, num_layers, dropout)
        self.gnn = to_hetero(base, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)


class BaseGAT(torch.nn.Module):
    """Base homogeneous GAT model."""

    def __init__(self, hidden_channels, out_channels, num_layers=2,
                 heads=4, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv((-1, -1), hidden_channels, heads=heads,
                                  add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv((-1, -1), hidden_channels, heads=heads,
                                      add_self_loops=False))
        self.convs.append(GATConv((-1, -1), out_channels, heads=1, concat=False,
                                  add_self_loops=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATBaseline(torch.nn.Module):
    """
    Heterogeneous GAT baseline.

    Uses GATConv layers with multi-head attention, converted to
    heterogeneous via to_hetero().
    """

    def __init__(self, hidden_channels, out_channels, metadata,
                 num_layers=2, heads=4, dropout=0.3, aggr='mean'):
        super().__init__()
        base = BaseGAT(hidden_channels, out_channels, num_layers, heads, dropout)
        self.gnn = to_hetero(base, metadata, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)
