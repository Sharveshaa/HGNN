import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear

class RGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, num_heads=4):
        super().__init__()
        self.conv = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels // num_heads, heads=num_heads, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.lin = nn.ModuleDict({
            node_type: Linear(-1, out_channels)
            for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.conv(x_dict, edge_index_dict)
        for node_type, x in x_dict.items():
            if node_type not in out_dict:
                out_dict[node_type] = self.lin[node_type](x)
            else:
                out_dict[node_type] = out_dict[node_type] + self.lin[node_type](x)
        return out_dict

class PhishingHGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, num_heads=4):
        super().__init__()
        self.conv1 = RGATLayer(-1, hidden_channels, metadata, num_heads)
        self.conv2 = RGATLayer(hidden_channels, hidden_channels, metadata, num_heads)
        self.conv3 = RGATLayer(hidden_channels, hidden_channels, metadata, num_heads)
        
        self.conv4 = HeteroConv({
            edge_type: GATConv((-1, -1), out_channels, heads=1, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='mean')
        self.lin_final = nn.ModuleDict({
            node_type: Linear(-1, out_channels)
            for node_type in metadata[0]
        })

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_dict, edge_index_dict, return_embeds=False):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        embeds = x_dict
        
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        out_dict = self.conv4(x_dict, edge_index_dict)
        for node_type, x in x_dict.items():
            if node_type not in out_dict:
                out_dict[node_type] = self.lin_final[node_type](x)
            else:
                out_dict[node_type] = out_dict[node_type] + self.lin_final[node_type](x)

        if return_embeds:
            return out_dict, embeds
        return out_dict
