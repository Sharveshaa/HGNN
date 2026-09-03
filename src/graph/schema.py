"""
Schema Registry for the HGNN Campaign Detection System.

Central source of truth for all node types, edge types, feature dimensions,
and meta-paths used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# Node Type Definitions
# ──────────────────────────────────────────────────────────────

@dataclass
class NodeTypeSpec:
    """Specification for a single node type."""
    name: str
    feature_dim: int
    description: str
    has_label: bool = True
    num_classes: int = 2  # default binary (benign vs malicious)


# Feature dimension breakdown:
#   user:     bio_emb(16) + numeric(10) + behavioral(6) = 32
#   post:     text_emb(16) + sentiment(1) + urgency(1) + clickbait(1) +
#             toxicity(1) + emotion(8) + topic(16) + numeric(4) = 48
#   url:      existing(6) + entropy_ratio(1) + char_patterns(7) = 14
#   domain:   domain_age, days_to_expiry, has_mx, has_ssl = 4
#   ip:       identity feature = 1
#   asn:      identity feature = 1
#   registrar: identity feature = 1
#   campaign: aggregated from member nodes = 16

NODE_TYPES: Dict[str, NodeTypeSpec] = {
    'user': NodeTypeSpec(
        name='user',
        feature_dim=32,
        description='Social media user profiles with bio embeddings and behavioral features',
        has_label=True,
        num_classes=2,
    ),
    'post': NodeTypeSpec(
        name='post',
        feature_dim=48,
        description='Social media posts with text embeddings, sentiment, and NLP features',
        has_label=True,
        num_classes=2,
    ),
    'url': NodeTypeSpec(
        name='url',
        feature_dim=14,
        description='URLs with structural and character-level features',
        has_label=True,
        num_classes=2,
    ),
    'domain': NodeTypeSpec(
        name='domain',
        feature_dim=4,
        description='Web domains with WHOIS and DNS features',
        has_label=False,
    ),
    'ip': NodeTypeSpec(
        name='ip',
        feature_dim=1,
        description='IP addresses (structural hub nodes)',
        has_label=False,
    ),
    'asn': NodeTypeSpec(
        name='asn',
        feature_dim=1,
        description='Autonomous System Numbers (structural hub nodes)',
        has_label=False,
    ),
    'registrar': NodeTypeSpec(
        name='registrar',
        feature_dim=1,
        description='Domain registrars (structural hub nodes)',
        has_label=False,
    ),
    'campaign': NodeTypeSpec(
        name='campaign',
        feature_dim=16,
        description='Campaign meta-nodes aggregated from member user/post nodes',
        has_label=True,
        num_classes=3,  # benign, suspicious, malicious
    ),
}


# ──────────────────────────────────────────────────────────────
# Edge Type Definitions
# ──────────────────────────────────────────────────────────────

@dataclass
class EdgeTypeSpec:
    """Specification for a single edge type."""
    src_type: str
    relation: str
    dst_type: str
    description: str
    is_temporal: bool = False
    is_inferred: bool = False  # True for edges computed via heuristics

    @property
    def triplet(self) -> Tuple[str, str, str]:
        return (self.src_type, self.relation, self.dst_type)


EDGE_TYPES: Dict[str, EdgeTypeSpec] = {
    # ── Social edges ──
    'user_follows_user': EdgeTypeSpec(
        'user', 'follows', 'user',
        'Social follow relationship',
    ),
    'user_posts_post': EdgeTypeSpec(
        'user', 'posts', 'post',
        'User authored a post',
    ),
    'user_shares_post': EdgeTypeSpec(
        'user', 'shares', 'post',
        'User reshared/amplified a post',
    ),
    'user_mentions_user': EdgeTypeSpec(
        'user', 'mentions', 'user',
        'Coordinated @-mention patterns between users',
        is_inferred=True,
    ),

    # ── Content edges ──
    'post_contains_url': EdgeTypeSpec(
        'post', 'contains', 'url',
        'Post contains a URL payload',
    ),
    'post_similar_to_post': EdgeTypeSpec(
        'post', 'similar_to', 'post',
        'Content similarity edge (cosine > threshold)',
        is_inferred=True,
    ),

    # ── Infrastructure edges ──
    'url_hosted_on_domain': EdgeTypeSpec(
        'url', 'hosted_on', 'domain',
        'URL hosted on a registered domain',
    ),
    'domain_resolves_to_ip': EdgeTypeSpec(
        'domain', 'resolves_to', 'ip',
        'Domain DNS resolution to IP',
    ),
    'domain_registered_via_registrar': EdgeTypeSpec(
        'domain', 'registered_via', 'registrar',
        'Domain registration through registrar',
    ),
    'ip_belongs_to_asn': EdgeTypeSpec(
        'ip', 'belongs_to', 'asn',
        'IP belongs to an Autonomous System',
    ),

    # ── Campaign edges ──
    'user_belongs_to_campaign': EdgeTypeSpec(
        'user', 'member_of', 'campaign',
        'User is a member of a detected campaign',
    ),
    'post_belongs_to_campaign': EdgeTypeSpec(
        'post', 'part_of', 'campaign',
        'Post is part of a detected campaign',
    ),
}


# ──────────────────────────────────────────────────────────────
# Meta-paths for HAN
# ──────────────────────────────────────────────────────────────

@dataclass
class MetaPath:
    """A meta-path definition for HAN semantic attention."""
    name: str
    node_types: List[str]  # sequence of node types in the path
    description: str

    @property
    def length(self) -> int:
        return len(self.node_types) - 1


# User-centric meta-paths
USER_META_PATHS = [
    MetaPath('UPU', ['user', 'post', 'user'],
             'Users connected through shared post authorship/sharing'),
    MetaPath('UUU', ['user', 'user', 'user'],
             'Users connected through follow chains'),
    MetaPath('UPUU', ['user', 'post', 'url', 'url'],
             'Users connected through posts sharing similar URLs'),
    MetaPath('UCU', ['user', 'campaign', 'user'],
             'Users connected through campaign co-membership'),
]

# Post-centric meta-paths
POST_META_PATHS = [
    MetaPath('PUP', ['post', 'user', 'post'],
             'Posts connected through the same author'),
    MetaPath('PUDP', ['post', 'url', 'domain', 'post'],
             'Posts connected through URLs on the same domain'),
    MetaPath('PCP', ['post', 'campaign', 'post'],
             'Posts connected through campaign co-membership'),
]

ALL_META_PATHS = USER_META_PATHS + POST_META_PATHS


# ──────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────

def get_metadata() -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Get the metadata tuple (node_types, edge_types) in PyG format.
    Used for to_hetero() and HGTConv initialization.
    """
    node_type_names = list(NODE_TYPES.keys())
    edge_triplets = [spec.triplet for spec in EDGE_TYPES.values()]
    return (node_type_names, edge_triplets)


def get_labeled_node_types() -> List[str]:
    """Return node types that have classification labels."""
    return [name for name, spec in NODE_TYPES.items() if spec.has_label]


def get_feature_dims() -> Dict[str, int]:
    """Return a mapping of node_type -> feature_dimension."""
    return {name: spec.feature_dim for name, spec in NODE_TYPES.items()}


def validate_graph(data) -> bool:
    """
    Validate that a HeteroData object conforms to the schema.
    Returns True if valid, raises ValueError otherwise.
    """
    errors = []

    # Check node types exist
    for ntype in data.node_types:
        if ntype not in NODE_TYPES:
            errors.append(f"Unknown node type: {ntype}")

    # Check feature dimensions
    for ntype in data.node_types:
        if ntype in NODE_TYPES and hasattr(data[ntype], 'x'):
            actual_dim = data[ntype].x.shape[1]
            expected_dim = NODE_TYPES[ntype].feature_dim
            if actual_dim != expected_dim:
                errors.append(
                    f"Node type '{ntype}': expected feature dim {expected_dim}, "
                    f"got {actual_dim}"
                )

    # Check edge types
    for etype in data.edge_types:
        triplet_found = any(
            spec.triplet == etype for spec in EDGE_TYPES.values()
        )
        if not triplet_found:
            errors.append(f"Unknown edge type: {etype}")

    # Check edge index shapes
    for etype in data.edge_types:
        ei = data[etype].edge_index
        if ei.dim() != 2 or ei.shape[0] != 2:
            errors.append(f"Edge type {etype}: invalid edge_index shape {ei.shape}")

    if errors:
        raise ValueError(
            "Graph validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True


if __name__ == '__main__':
    print("=== HGNN Schema Registry ===\n")

    print("Node Types:")
    for name, spec in NODE_TYPES.items():
        label_info = f", {spec.num_classes} classes" if spec.has_label else ", no label"
        print(f"  {name:12s} dim={spec.feature_dim:3d}{label_info}")

    print("\nEdge Types:")
    for name, spec in EDGE_TYPES.items():
        flags = []
        if spec.is_temporal:
            flags.append("temporal")
        if spec.is_inferred:
            flags.append("inferred")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  ({spec.src_type}, {spec.relation}, {spec.dst_type}){flag_str}")

    print(f"\nMeta-paths: {len(ALL_META_PATHS)}")
    for mp in ALL_META_PATHS:
        print(f"  {mp.name}: {' → '.join(mp.node_types)}")

    print(f"\nPyG metadata: {get_metadata()}")
