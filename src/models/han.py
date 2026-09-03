"""
Heterogeneous Attention Network (HAN) for the HGNN Campaign Detection System.

Implements HAN with:
    1. Per-meta-path attention: GAT-style attention within each meta-path subgraph
    2. Semantic-level attention: Attention across meta-path-specific embeddings
    3. Per-type classifier heads with MLP

Reference: Wang et al., "Heterogeneous Graph Attention Network" (WWW 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear
from typing import Dict, List, Tuple, Optional


class MetaPathAttention(nn.Module):
    """
    Computes attention-based aggregation along a single meta-path.

    Uses GATConv to aggregate neighbor information along the meta-path
    adjacency, producing meta-path-specific node embeddings.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.gat = GATConv(
            in_channels, out_channels, heads=heads,
            dropout=dropout, add_self_loops=True, concat=False,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Meta-path adjacency [2, E]

        Returns:
            Meta-path embeddings [N, out_channels]
        """
        if edge_index.numel() == 0:
            # No edges for this meta-path — return projected features
            return self.norm(x[:, :self.gat.out_channels] if x.shape[1] > self.gat.out_channels
                           else F.pad(x, (0, self.gat.out_channels - x.shape[1])))

        out = self.gat(x, edge_index)
        return self.norm(out)


class SemanticAttention(nn.Module):
    """
    Learns attention weights across different meta-path embeddings.

    Given K meta-path-specific embeddings for each node, computes
    a weighted sum using learned attention coefficients.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(self, z_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z_list: List of K tensors, each [N, hidden_dim]

        Returns:
            Weighted sum [N, hidden_dim]
        """
        if len(z_list) == 1:
            return z_list[0]

        # Stack: [K, N, hidden_dim]
        z_stack = torch.stack(z_list, dim=0)

        # Compute attention scores: [K, N, 1]
        attn_scores = self.project(z_stack)

        # Normalize across meta-paths: [K, N, 1]
        attn_weights = F.softmax(attn_scores, dim=0)

        # Weighted sum: [N, hidden_dim]
        output = (attn_weights * z_stack).sum(dim=0)

        return output


class HANLayer(nn.Module):
    """
    A single HAN layer combining meta-path attention and semantic attention.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        meta_paths: List[Tuple[str, str]],  # List of (meta_path_name, target_node_type)
        heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels_dict: {node_type: feature_dim}
            out_channels: Output dimension for each meta-path
            meta_paths: List of (meta_path_name, target_node_type) tuples
            heads: Number of GAT attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.meta_paths = meta_paths
        self.out_channels = out_channels

        # Type-specific linear projections to shared dimension
        self.projections = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.projections[ntype] = Linear(in_dim, out_channels)

        # Per-meta-path attention modules
        self.meta_path_attns = nn.ModuleDict()
        for mp_name, _ in meta_paths:
            self.meta_path_attns[mp_name] = MetaPathAttention(
                out_channels, out_channels, heads=heads, dropout=dropout
            )

        # Semantic attention (across meta-paths)
        self.semantic_attn = SemanticAttention(out_channels)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        meta_path_edge_indices: Dict[str, torch.Tensor],
        target_node_type: str,
    ) -> torch.Tensor:
        """
        Args:
            x_dict: {node_type: features}
            meta_path_edge_indices: {meta_path_name: edge_index [2, E]}
            target_node_type: The node type to produce embeddings for

        Returns:
            Embeddings for target_node_type [N, out_channels]
        """
        # Project all node types to shared dimension
        projected = {}
        for ntype, x in x_dict.items():
            if ntype in self.projections:
                projected[ntype] = self.projections[ntype](x)
            else:
                projected[ntype] = x

        # Get target node features
        target_x = projected.get(target_node_type)
        if target_x is None:
            target_x = x_dict[target_node_type]

        # Compute meta-path-specific embeddings
        z_list = []
        for mp_name, mp_target in self.meta_paths:
            if mp_target != target_node_type:
                continue

            edge_index = meta_path_edge_indices.get(mp_name)
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long,
                                        device=target_x.device)

            z = self.meta_path_attns[mp_name](target_x, edge_index)
            z_list.append(z)

        if not z_list:
            return target_x

        # Semantic attention across meta-paths
        return self.semantic_attn(z_list)


class HAN(nn.Module):
    """
    Heterogeneous Attention Network (HAN).

    Architecture:
        1. Type-specific linear projections
        2. Per-meta-path GAT attention
        3. Semantic attention across meta-paths
        4. Per-type MLP classifier heads

    The model computes meta-path adjacency matrices on-the-fly from
    the heterogeneous edge indices, or accepts pre-computed ones.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        out_channels_dict: Optional[Dict[str, int]] = None,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels_dict: {node_type: input_feature_dim}
            hidden_channels: Hidden dimension
            out_channels_dict: {node_type: num_classes} for classification
            num_layers: Number of HAN layers
            heads: Number of attention heads per meta-path
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        if out_channels_dict is None:
            out_channels_dict = {nt: 2 for nt in in_channels_dict}
        self.out_channels_dict = out_channels_dict

        # Define meta-paths with their target node types
        self.meta_path_defs = [
            ('UPU', 'user'),     # user → post → user
            ('UUU', 'user'),     # user → user → user (follow chains)
            ('UCU', 'user'),     # user → campaign → user
            ('PUP', 'post'),     # post → user → post
            ('PCP', 'post'),     # post → campaign → post
        ]

        # Input projection to hidden dim
        self.input_proj = nn.ModuleDict()
        for ntype, in_dim in in_channels_dict.items():
            self.input_proj[ntype] = Linear(in_dim, hidden_channels)

        # HAN layers
        all_same_dim = {nt: hidden_channels for nt in in_channels_dict}
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = HANLayer(
                in_channels_dict=all_same_dim,
                out_channels=hidden_channels,
                meta_paths=self.meta_path_defs,
                heads=heads,
                dropout=dropout,
            )
            self.layers.append(layer)

        # Per-type classifier heads
        self.classifiers = nn.ModuleDict()
        for ntype, n_classes in out_channels_dict.items():
            self.classifiers[ntype] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, n_classes),
            )

    def compute_meta_path_adjacency(
        self,
        edge_index_dict: Dict[Tuple, torch.Tensor],
        num_nodes_dict: Dict[str, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute meta-path adjacency matrices from heterogeneous edges.

        For meta-path A-B-A: adj = (A→B) @ (B→A)^T
        Returns sparse edge indices for each meta-path.
        """
        meta_path_edges = {}

        def get_edge(src_type, rel, dst_type):
            """Find edge index for a given triplet."""
            for etype, ei in edge_index_dict.items():
                if etype == (src_type, rel, dst_type):
                    return ei
            return None

        def compose_paths(ei_ab, ei_bc, n_a, n_b, n_c):
            """Compose two edge indices to get A→C via B."""
            if ei_ab is None or ei_bc is None:
                return torch.empty((2, 0), dtype=torch.long)
            if ei_ab.numel() == 0 or ei_bc.numel() == 0:
                return torch.empty((2, 0), dtype=torch.long)

            # Build B→C adjacency as dict
            b_to_c = {}
            for i in range(ei_bc.shape[1]):
                b = ei_bc[0, i].item()
                c = ei_bc[1, i].item()
                if b not in b_to_c:
                    b_to_c[b] = []
                b_to_c[b].append(c)

            # Compose: for each A→B edge, find B→C edges
            src_list, dst_list = [], []
            for i in range(ei_ab.shape[1]):
                a = ei_ab[0, i].item()
                b = ei_ab[1, i].item()
                if b in b_to_c:
                    for c in b_to_c[b]:
                        if a != c:  # Remove self-loops
                            src_list.append(a)
                            dst_list.append(c)

            if not src_list:
                return torch.empty((2, 0), dtype=torch.long)

            # Deduplicate
            edges = set(zip(src_list, dst_list))
            src_list = [e[0] for e in edges]
            dst_list = [e[1] for e in edges]

            device = ei_ab.device
            return torch.tensor([src_list, dst_list], dtype=torch.long, device=device)

        # UPU: user → post → user (via authorship/sharing)
        ei_up = get_edge('user', 'posts', 'post')
        if ei_up is None:
            ei_up = get_edge('user', 'shares', 'post')
        # Reverse edge: post → user
        ei_pu = None
        if ei_up is not None and ei_up.numel() > 0:
            ei_pu = torch.stack([ei_up[1], ei_up[0]])
        n_u = num_nodes_dict.get('user', 0)
        n_p = num_nodes_dict.get('post', 0)
        meta_path_edges['UPU'] = compose_paths(ei_up, ei_pu, n_u, n_p, n_u)

        # UUU: user → user → user (follow chains)
        ei_uu = get_edge('user', 'follows', 'user')
        meta_path_edges['UUU'] = compose_paths(ei_uu, ei_uu, n_u, n_u, n_u)

        # UCU: user → campaign → user
        ei_uc = get_edge('user', 'member_of', 'campaign')
        ei_cu = None
        if ei_uc is not None and ei_uc.numel() > 0:
            ei_cu = torch.stack([ei_uc[1], ei_uc[0]])
        n_c = num_nodes_dict.get('campaign', 0)
        meta_path_edges['UCU'] = compose_paths(ei_uc, ei_cu, n_u, n_c, n_u)

        # PUP: post → user → post
        if ei_pu is not None:
            ei_up2 = get_edge('user', 'posts', 'post')
            meta_path_edges['PUP'] = compose_paths(ei_pu, ei_up2, n_p, n_u, n_p)
        else:
            meta_path_edges['PUP'] = torch.empty((2, 0), dtype=torch.long)

        # PCP: post → campaign → post
        ei_pc = get_edge('post', 'part_of', 'campaign')
        ei_cp = None
        if ei_pc is not None and ei_pc.numel() > 0:
            ei_cp = torch.stack([ei_pc[1], ei_pc[0]])
        meta_path_edges['PCP'] = compose_paths(ei_pc, ei_cp, n_p, n_c, n_p)

        return meta_path_edges

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        meta_path_edges: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_dict: {node_type: features}
            edge_index_dict: {(src, rel, dst): edge_index}
            meta_path_edges: Optional pre-computed meta-path adjacencies

        Returns:
            {node_type: logits} for classified node types
        """
        # Compute meta-path adjacency if not provided
        if meta_path_edges is None:
            num_nodes_dict = {nt: x.shape[0] for nt, x in x_dict.items()}
            meta_path_edges = self.compute_meta_path_adjacency(
                edge_index_dict, num_nodes_dict
            )

        # Project to hidden dimension
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.input_proj:
                h_dict[ntype] = F.relu(self.input_proj[ntype](x))
                h_dict[ntype] = F.dropout(h_dict[ntype], p=self.dropout,
                                          training=self.training)
            else:
                h_dict[ntype] = x

        # Apply HAN layers
        for layer in self.layers:
            new_h = {}
            for ntype in h_dict:
                if any(mp_target == ntype for _, mp_target in self.meta_path_defs):
                    new_h[ntype] = layer(h_dict, meta_path_edges, ntype)
                else:
                    new_h[ntype] = h_dict[ntype]
            h_dict = new_h

        # Classify
        out_dict = {}
        for ntype in self.out_channels_dict:
            if ntype in h_dict and ntype in self.classifiers:
                out_dict[ntype] = self.classifiers[ntype](h_dict[ntype])

        return out_dict
