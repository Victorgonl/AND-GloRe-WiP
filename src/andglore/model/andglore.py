from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

from andglore.model.attention import Attention
from andglore.model.gat import GAT


class ANDGloRe(nn.Module):
    """
    Author Name Disambiguation via Global and Refined Views (AND-GloRe).
    """

    def __init__(
        self,
        num_papers,
        num_authors,
        num_venues,
        num_orgs,
        input_dim,
        hidden_dim,
        embedding_dim,
        projection_dim,
        network_encoder_hidden_dim,
        temperature,
        dropout,
        network_encoder_dropout,
        network_encoder_alpha,
    ):
        super(ANDGloRe, self).__init__()

        # Internal parameters
        self.temperature = temperature

        # Hold embeddings for inference
        self.global_embeddings = None
        self.refined_embeddings = None
        self.pap_embeddings = None
        self.pvp_embeddings = None
        self.pop_embeddings = None
        self.paoap_embeddings = None

        # Layers
        ## Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        ## Feature projection
        self.feature_projection = nn.Linear(input_dim, hidden_dim, bias=True)
        # Identity feature matrices.
        self.author_projection = nn.Linear(num_authors, hidden_dim, bias=True)
        self.venue_projection = nn.Linear(num_venues, hidden_dim, bias=True)
        self.org_projection = nn.Linear(num_orgs, hidden_dim, bias=True)
        ## Metapath projections
        self.pap_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pvp_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pop_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.paoap_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        ## Shared network encoder
        self.network_encoder = GAT(
            hidden_dim,
            network_encoder_hidden_dim,
            embedding_dim,
            network_encoder_dropout,
            alpha=network_encoder_alpha,
        )
        ## Metapath attention
        self.metapath_attention = Attention(embedding_dim, dropout)
        ## Contrastive projection layer
        self.contrastive_projection = nn.Linear(embedding_dim, projection_dim)
        ## Weighted InfoNCE loss MLP
        self.infonce_mlp = nn.Sequential(
            nn.Linear(projection_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def project_identity_features(self, projection):
        features = projection.weight.t()
        if projection.bias is not None:
            features = features + projection.bias
        return F.elu(self.dropout_layer(features))

    def aggregate_metapath(
        self,
        paper_features,
        aug_paper_features,
        neighbor_features,
        neighbor_incidence,
        metapath_projection,
    ):
        neighbor_counts = neighbor_incidence.sum(dim=1, keepdim=True)
        neighbor_counts = neighbor_counts.clamp_min(1.0)

        neighbor_context = torch.mm(
            neighbor_incidence,
            neighbor_features,
        )
        neighbor_context = neighbor_context / neighbor_counts

        original_features = F.elu(
            paper_features + self.dropout_layer(metapath_projection(neighbor_context))
        )

        augmented_features = F.elu(
            aug_paper_features
            + self.dropout_layer(metapath_projection(neighbor_context))
        )

        return original_features, augmented_features

    def forward(
        self,
        paper_features,
        bipartite_pa,
        bipartite_pv,
        bipartite_po,
        bipartite_ao,
        bipartite_pao,
        pp_adj_global,
        pp_adj_pap,
        pp_adj_pvp,
        pp_adj_pop,
        pp_adj_paoap,
        aug_paper_features,
        aug_pp_adj_pap,
        aug_pp_adj_pvp,
        aug_pp_adj_pop,
        aug_pp_adj_paoap,
    ):

        # Project features
        ## Original paper features
        paper_features = F.elu(
            self.dropout_layer(self.feature_projection(paper_features))
        )
        ## Augmented paper features
        aug_paper_features = F.elu(
            self.dropout_layer(self.feature_projection(aug_paper_features))
        )
        ## Intermediate node features
        author_features = self.project_identity_features(self.author_projection)
        venue_features = self.project_identity_features(self.venue_projection)
        org_features = self.project_identity_features(self.org_projection)

        # Global view
        global_embeddings = self.network_encoder(
            paper_features,
            pp_adj_global,
        )

        # Refined view
        ## P-A-P features
        pap_features, aug_pap_features = self.aggregate_metapath(
            paper_features,
            aug_paper_features,
            author_features,
            bipartite_pa,
            self.pap_projection,
        )
        ## P-V-P features
        pvp_features, aug_pvp_features = self.aggregate_metapath(
            paper_features,
            aug_paper_features,
            venue_features,
            bipartite_pv,
            self.pvp_projection,
        )
        ## P-O-P features
        pop_features, aug_pop_features = self.aggregate_metapath(
            paper_features,
            aug_paper_features,
            org_features,
            bipartite_po,
            self.pop_projection,
        )
        ## P-A-O-A-P features
        paoap_features, aug_paoap_features = self.aggregate_metapath(
            paper_features,
            aug_paper_features,
            org_features,
            bipartite_pao,
            self.paoap_projection,
        )

        # Encode metapath views
        ## P-A-P views
        pap_embeddings = self.network_encoder(
            pap_features,
            pp_adj_pap,
        )
        aug_pap_embeddings = self.network_encoder(
            aug_pap_features,
            aug_pp_adj_pap,
        )
        ## P-V-P views
        pvp_embeddings = self.network_encoder(
            pvp_features,
            pp_adj_pvp,
        )
        aug_pvp_embeddings = self.network_encoder(
            aug_pvp_features,
            aug_pp_adj_pvp,
        )
        ## P-O-P views
        pop_embeddings = self.network_encoder(
            pop_features,
            pp_adj_pop,
        )
        aug_pop_embeddings = self.network_encoder(
            aug_pop_features,
            aug_pp_adj_pop,
        )
        ## P-A-O-A-P views
        paoap_embeddings = self.network_encoder(
            paoap_features,
            pp_adj_paoap,
        )
        aug_paoap_embeddings = self.network_encoder(
            aug_paoap_features,
            aug_pp_adj_paoap,
        )

        # Normalize metapath embeddings
        pap_embeddings = F.normalize(pap_embeddings, dim=1)
        aug_pap_embeddings = F.normalize(aug_pap_embeddings, dim=1)
        pvp_embeddings = F.normalize(pvp_embeddings, dim=1)
        aug_pvp_embeddings = F.normalize(aug_pvp_embeddings, dim=1)
        pop_embeddings = F.normalize(pop_embeddings, dim=1)
        aug_pop_embeddings = F.normalize(aug_pop_embeddings, dim=1)
        paoap_embeddings = F.normalize(paoap_embeddings, dim=1)
        aug_paoap_embeddings = F.normalize(
            aug_paoap_embeddings,
            dim=1,
        )

        # Refined embeddings via metapath attention
        refined_embeddings = self.metapath_attention(
            [
                pap_embeddings,
                aug_pap_embeddings,
                pvp_embeddings,
                aug_pvp_embeddings,
                pop_embeddings,
                aug_pop_embeddings,
                paoap_embeddings,
                aug_paoap_embeddings,
            ]
        )

        # Hold embeddings for inference
        self.global_embeddings = global_embeddings.clone().detach().cpu()
        self.refined_embeddings = refined_embeddings.clone().detach().cpu()
        self.pap_embeddings = pap_embeddings.clone().detach().cpu()
        self.pvp_embeddings = pvp_embeddings.clone().detach().cpu()
        self.pop_embeddings = pop_embeddings.clone().detach().cpu()
        self.paoap_embeddings = paoap_embeddings.clone().detach().cpu()

        # Projection for contrastive learning
        ## Global view
        global_embeddings = torch.tanh(self.contrastive_projection(global_embeddings))
        global_embeddings = F.normalize(global_embeddings, dim=1)
        ## Refined view
        refined_embeddings = torch.tanh(self.contrastive_projection(refined_embeddings))
        refined_embeddings = F.normalize(refined_embeddings, dim=1)

        # Contrastive loss
        loss = self.weighted_infonce(
            refined_embeddings,
            global_embeddings,
        )
        return loss

    def weighted_infonce(self, z_refined, z_global):
        N = z_global.size(0)

        sim = torch.mm(z_refined, z_global.t()) / self.temperature
        dots = torch.exp(sim)
        nominator = dots.diagonal()

        pair_sum = z_refined.unsqueeze(1) + z_global.unsqueeze(0)  # (N, N, D)
        weight = self.infonce_mlp(pair_sum.reshape(N * N, -1)).reshape(N, N)

        neg = dots * weight
        denominator = neg.sum(dim=1)

        loss = -torch.log(nominator / denominator).mean()
        return loss
