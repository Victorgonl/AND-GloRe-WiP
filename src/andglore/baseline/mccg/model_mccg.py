import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayerMCCG(nn.Module):
    def __init__(self, in_features: int, out_features: int, alpha: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.attention_self = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.xavier_uniform_(self.attention_self.data, gain=1.414)
        self.attention_neighbors = nn.Parameter(torch.zeros(out_features, 1))
        nn.init.xavier_uniform_(self.attention_neighbors.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(
        self,
        inputs: torch.Tensor,
        adjacency: torch.Tensor,
        diffusion: torch.Tensor,
        concat: bool = True,
    ) -> torch.Tensor:
        hidden = inputs @ self.weight
        attention = hidden @ self.attention_self
        attention = attention + (hidden @ self.attention_neighbors).t()
        attention = self.leaky_relu(attention * diffusion)
        masked_attention = torch.where(
            adjacency > 0,
            attention,
            torch.full_like(adjacency, -9e15),
        )
        output = F.softmax(masked_attention, dim=1) @ hidden
        return F.elu(output) if concat else output


class GATMCCG(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        alpha: float = 0.2,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.conv1_mccg = GATLayerMCCG(input_dim, hidden_dim, alpha)
        self.conv2_mccg = GATLayerMCCG(hidden_dim, output_dim, alpha)
        self.dropout_mccg = dropout

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor, diffusion: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.conv1_mccg(features, adjacency, diffusion)
        hidden = F.dropout(hidden, self.dropout_mccg, training=self.training)
        return self.conv2_mccg(hidden, adjacency, diffusion)


class MCCG(nn.Module):
    def __init__(
        self,
        encoder_mccg: GATMCCG,
        hidden_dim: int,
        multiview_projection_dim: int,
        cluster_projection_dim: int,
    ) -> None:
        super().__init__()
        self.encoder_mccg = encoder_mccg
        self.multiview_projector_mccg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, multiview_projection_dim),
        )
        self.cluster_projector_mccg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, cluster_projection_dim),
        )
        self.pair_projector_mccg = nn.Linear(cluster_projection_dim, 32)
        self.pair_weight_mccg = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features_view1: torch.Tensor,
        adjacency_view1: torch.Tensor,
        diffusion_view1: torch.Tensor,
        features_view2: torch.Tensor,
        adjacency_view2: torch.Tensor,
        diffusion_view2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_view1 = self.encoder_mccg(
            features_view1, adjacency_view1, diffusion_view1
        )
        embedding_view2 = self.encoder_mccg(
            features_view2, adjacency_view2, diffusion_view2
        )
        embedding = (embedding_view1 + embedding_view2) / 2

        projected_view1 = F.normalize(
            self.multiview_projector_mccg(embedding_view1), dim=1
        )
        projected_view2 = F.normalize(
            self.multiview_projector_mccg(embedding_view2), dim=1
        )
        cluster_embedding = F.normalize(self.cluster_projector_mccg(embedding), dim=1)
        multiview_embedding = torch.cat(
            [projected_view1.unsqueeze(1), projected_view2.unsqueeze(1)], dim=1
        )
        return multiview_embedding, cluster_embedding

    def self_supervised_contrastive_loss_mccg(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        temperature: float = 0.2,
        contrast_mode: str = "all",
    ) -> torch.Tensor:
        device = features.device
        if features.ndim < 3:
            raise ValueError("features must have shape [batch, views, ...]")
        if features.ndim > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both labels and mask")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match number of features")
            mask = torch.eq(labels, labels.t()).float().to(device)
        else:
            mask = mask.float().to(device)  # type: ignore[union-attr]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast mode: {contrast_mode}")

        logits = (anchor_feature @ contrast_feature.t()) / temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask
        exp_logits = exp_logits * logits_mask

        if contrast_mode == "one":
            weights = self.pair_projector_mccg(anchor_feature)
            count = weights.size(0)
            pair_features = (weights.unsqueeze(1) + weights.unsqueeze(0)).reshape(
                count * count, -1
            )
            pair_weights = self.pair_weight_mccg(pair_features).reshape(count, count)
            exp_logits = exp_logits * (pair_weights / temperature)
            logits = exp_logits

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(1e-12))
        positive_pairs = mask.sum(1).clamp_min(1.0)
        mean_log_prob_positive = (mask * log_prob).sum(1) / positive_pairs
        loss = -mean_log_prob_positive / temperature
        return loss.view(anchor_count, batch_size).mean()

    SelfSupConLoss = self_supervised_contrastive_loss_mccg
