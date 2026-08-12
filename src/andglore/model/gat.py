import torch
import torch.linalg as torch_linalg
import torch.nn.functional as F
from torch import nn


def get_proximity_matrix(adjacency_matrix, step_count=2, exclude_self_loops=False):
    """
    Average proximity-probability matrix powers up to ``step_count``.
    """

    transition_probabilities = F.normalize(adjacency_matrix, p=1, dim=1)

    proximity_matrix = (
        sum(
            torch_linalg.matrix_power(transition_probabilities, step)
            for step in range(1, step_count + 1)
        )
        / step_count
    )

    if exclude_self_loops:
        proximity_matrix.fill_diagonal_(0)  # type: ignore

    return proximity_matrix


class GATLayer(nn.Module):
    """
    Single graph attention layer with optional transition-prior weighting.
    """

    def __init__(self, input_dim, output_dim, alpha):
        """
        Initialize trainable feature projection and attention vectors.
        """

        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.feature_weight_matrix = nn.Parameter(
            torch.zeros(size=(input_dim, output_dim))
        )
        nn.init.xavier_uniform_(self.feature_weight_matrix.data, gain=1.414)

        self.self_attention_vector = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        nn.init.xavier_uniform_(self.self_attention_vector.data, gain=1.414)

        self.neighbor_attention_vector = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        nn.init.xavier_uniform_(self.neighbor_attention_vector.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(
        self,
        features,
        adjacency_matrix,
        proximity_matrix=None,
        apply_activation=True,
    ):
        """
        Apply attention over adjacent nodes and return transformed features.
        """

        projected_features = torch.mm(features, self.feature_weight_matrix)

        self_attention_scores = torch.mm(projected_features, self.self_attention_vector)
        neighbor_attention_scores = torch.mm(
            projected_features, self.neighbor_attention_vector
        )
        attention_logits = self_attention_scores + torch.transpose(
            neighbor_attention_scores, 0, 1
        )
        if proximity_matrix is not None:
            attention_logits = torch.mul(attention_logits, proximity_matrix)
        attention_logits = self.leakyrelu(attention_logits)

        # Use a bool mask (1 byte/elem) instead of a float zero_vec (4 bytes/elem).
        attention_logits = attention_logits.masked_fill(adjacency_matrix <= 0, -9e15)
        attention_weights = F.softmax(attention_logits, dim=1)
        output_features = torch.matmul(attention_weights, projected_features)

        if apply_activation:
            output_features = F.elu(output_features)

        return output_features


class GAT(nn.Module):
    """
    Graph Attention Network with Proximity (GAT) adapted for graph clustering tasks, proposed in the original paper *Attributed graph clustering: a deep attentional embedding approach* (https://arxiv.org/pdf/1906.06532).
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.2,
        alpha=0.2,
        use_proximity_matrix=True,
    ):
        """
        Initialize a two-layer GAT encoder.
        """

        super(GAT, self).__init__()
        self.dropout = dropout
        self.use_proximity_matrix = use_proximity_matrix
        self.first_attention_layer = GATLayer(input_dim, hidden_dim, alpha)
        self.second_attention_layer = GATLayer(hidden_dim, output_dim, alpha)

    def forward(self, features, adjacency_matrix):
        """
        Encode node features with two attention layers.
        """

        if self.use_proximity_matrix:
            proximity_matrix = get_proximity_matrix(adjacency_matrix)
        else:
            proximity_matrix = None

        hidden_features = self.first_attention_layer(
            features, adjacency_matrix, proximity_matrix
        )
        hidden_features = F.dropout(
            hidden_features, self.dropout, training=self.training
        )
        output_features = self.second_attention_layer(
            hidden_features, adjacency_matrix, proximity_matrix
        )

        return output_features
