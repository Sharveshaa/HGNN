import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class PhishingHGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata):
        super().__init__()
        # Use to_hetero to convert the homogeneous BaseGNN into a heterogeneous one
        self.gnn = to_hetero(BaseGNN(hidden_channels, out_channels), metadata, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)
