import torch
from torch import nn


class Attention(nn.Module):
    """
    Learn attention weights over embeddings.
    """

    def __init__(self, hidden_dim, attention_dropout_rate=0.0):
        """
        Initialize the attention projection and scoring parameters.
        """

        super(Attention, self).__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.projection.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.query = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.query.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        self.attention_dropout = (
            nn.Dropout(attention_dropout_rate)
            if attention_dropout_rate
            else lambda x: x
        )

    def forward(self, embeddings):
        """
        Combine embedding views into one weighted representation.
        """

        attention_scores = []
        query = self.attention_dropout(self.query)
        for embedding in embeddings:
            embedding_summary = self.tanh(self.projection(embedding)).mean(dim=0)
            attention_scores.append(query.matmul(embedding_summary.t()))

        attention_weights = torch.cat(attention_scores, dim=-1).view(-1)
        attention_weights = self.softmax(attention_weights)

        metapath_embedding = sum(
            embeddings[index] * attention_weights[index]
            for index in range(len(embeddings))
        )
        return metapath_embedding
