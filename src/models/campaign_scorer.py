"""
Campaign Risk Scoring for the HGNN Campaign Detection System.

Implements:
    1. CampaignNodeEncoder: Attention-pooled aggregation of member node embeddings
    2. CampaignRiskScorer: MLP classifier for campaign risk level
    3. Interpretable risk decomposition with per-factor contribution weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over a set of embeddings.

    Learns to weight member node embeddings based on their relevance
    to the aggregated campaign representation.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, max_members, hidden_dim] or [max_members, hidden_dim]
            mask: [batch_size, max_members] boolean mask (True = valid)

        Returns:
            Pooled embedding [batch_size, hidden_dim] or [hidden_dim]
        """
        squeeze = False
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze = True

        # Compute attention weights [batch_size, max_members, 1]
        attn_scores = self.attn(embeddings)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=1)

        # Handle all-masked case
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum [batch_size, hidden_dim]
        pooled = (attn_weights * embeddings).sum(dim=1)

        if squeeze:
            pooled = pooled.squeeze(0)

        return pooled


class CampaignNodeEncoder(nn.Module):
    """
    Encodes campaign nodes by aggregating member user/post embeddings
    using attention pooling.

    Produces a campaign embedding that captures the collective signal
    from all member entities.
    """

    def __init__(
        self,
        user_embed_dim: int,
        post_embed_dim: int,
        campaign_embed_dim: int = 128,
        max_members: int = 50,
    ):
        super().__init__()
        self.max_members = max_members
        self.campaign_embed_dim = campaign_embed_dim

        # Project user/post embeddings to shared dimension
        self.user_proj = nn.Linear(user_embed_dim, campaign_embed_dim)
        self.post_proj = nn.Linear(post_embed_dim, campaign_embed_dim)

        # Attention pooling for users and posts separately
        self.user_pool = AttentionPooling(campaign_embed_dim)
        self.post_pool = AttentionPooling(campaign_embed_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(campaign_embed_dim * 2, campaign_embed_dim),
            nn.ReLU(),
            nn.LayerNorm(campaign_embed_dim),
        )

    def forward(
        self,
        user_embeddings: torch.Tensor,
        post_embeddings: torch.Tensor,
        user_membership: torch.Tensor,
        post_membership: torch.Tensor,
        num_campaigns: int,
    ) -> torch.Tensor:
        """
        Args:
            user_embeddings: [N_users, user_embed_dim]
            post_embeddings: [N_posts, post_embed_dim]
            user_membership: [2, E_uc] edge index (user → campaign)
            post_membership: [2, E_pc] edge index (post → campaign)
            num_campaigns: Number of campaign nodes

        Returns:
            Campaign embeddings [num_campaigns, campaign_embed_dim]
        """
        device = user_embeddings.device

        # Project to shared dimension
        user_proj = F.relu(self.user_proj(user_embeddings))
        post_proj = F.relu(self.post_proj(post_embeddings))

        campaign_embeds = torch.zeros(
            num_campaigns, self.campaign_embed_dim, device=device
        )

        for c_idx in range(num_campaigns):
            # Gather member user embeddings
            if user_membership.numel() > 0:
                user_mask = user_membership[1] == c_idx
                member_user_indices = user_membership[0][user_mask]
                if len(member_user_indices) > 0:
                    member_user_embeds = user_proj[member_user_indices]
                    # Limit to max_members
                    if len(member_user_indices) > self.max_members:
                        member_user_embeds = member_user_embeds[:self.max_members]
                    user_pooled = self.user_pool(member_user_embeds)
                else:
                    user_pooled = torch.zeros(self.campaign_embed_dim, device=device)
            else:
                user_pooled = torch.zeros(self.campaign_embed_dim, device=device)

            # Gather member post embeddings
            if post_membership.numel() > 0:
                post_mask = post_membership[1] == c_idx
                member_post_indices = post_membership[0][post_mask]
                if len(member_post_indices) > 0:
                    member_post_embeds = post_proj[member_post_indices]
                    if len(member_post_indices) > self.max_members:
                        member_post_embeds = member_post_embeds[:self.max_members]
                    post_pooled = self.post_pool(member_post_embeds)
                else:
                    post_pooled = torch.zeros(self.campaign_embed_dim, device=device)
            else:
                post_pooled = torch.zeros(self.campaign_embed_dim, device=device)

            # Fuse user and post signals
            combined = torch.cat([user_pooled, post_pooled])
            campaign_embeds[c_idx] = self.fusion(combined)

        return campaign_embeds


class CampaignRiskScorer(nn.Module):
    """
    Campaign Risk Scoring MLP.

    Takes campaign embeddings + structural features and outputs:
        1. Risk class (benign/suspicious/malicious)
        2. Continuous risk score [0, 1]
        3. Per-factor contribution weights (for interpretability)

    Structural features:
        - num_members (users)
        - num_posts
        - temporal_velocity (posts per hour)
        - infra_concentration (unique_domains / total_urls)
        - avg_member_account_age
        - avg_urgency_score
    """

    NUM_STRUCTURAL_FEATURES = 6

    def __init__(
        self,
        campaign_embed_dim: int = 128,
        num_risk_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        input_dim = campaign_embed_dim + self.NUM_STRUCTURAL_FEATURES

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_risk_classes),
        )

        # Continuous risk score head
        self.risk_scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Factor attribution weights (for interpretability)
        self.factor_attention = nn.Sequential(
            nn.Linear(input_dim, self.NUM_STRUCTURAL_FEATURES),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        campaign_embeddings: torch.Tensor,
        structural_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            campaign_embeddings: [N_campaigns, campaign_embed_dim]
            structural_features: [N_campaigns, 6]

        Returns:
            Dict with keys:
                'logits': [N, num_risk_classes]
                'risk_score': [N, 1]
                'factor_weights': [N, 6]
        """
        combined = torch.cat([campaign_embeddings, structural_features], dim=-1)

        return {
            'logits': self.classifier(combined),
            'risk_score': self.risk_scorer(combined),
            'factor_weights': self.factor_attention(combined),
        }


class CampaignDetectionModule(nn.Module):
    """
    End-to-end campaign detection module that combines:
    1. CampaignNodeEncoder (attention pooling of member embeddings)
    2. CampaignRiskScorer (risk classification + scoring)

    This module can be plugged into any base GNN as a downstream
    task head for campaign-level predictions.
    """

    def __init__(
        self,
        user_embed_dim: int,
        post_embed_dim: int,
        campaign_embed_dim: int = 128,
        num_risk_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = CampaignNodeEncoder(
            user_embed_dim=user_embed_dim,
            post_embed_dim=post_embed_dim,
            campaign_embed_dim=campaign_embed_dim,
        )
        self.scorer = CampaignRiskScorer(
            campaign_embed_dim=campaign_embed_dim,
            num_risk_classes=num_risk_classes,
            dropout=dropout,
        )

    def forward(
        self,
        user_embeddings: torch.Tensor,
        post_embeddings: torch.Tensor,
        user_membership: torch.Tensor,
        post_membership: torch.Tensor,
        structural_features: torch.Tensor,
        num_campaigns: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode campaigns → score risk.
        """
        campaign_embeds = self.encoder(
            user_embeddings, post_embeddings,
            user_membership, post_membership,
            num_campaigns,
        )
        return self.scorer(campaign_embeds, structural_features)
