import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCN(nn.Module):
    """Configurable GCN with optional batchnorm, dropout and MLP readout.

    Args:
        num_node_features (int): input node feature size
        hidden_channels (int): hidden dimension
        num_layers (int): number of GCN layers (>=1)
        dropout (float): dropout probability after activations
        use_batchnorm (bool): whether to use BatchNorm1d after each conv
    """

    def __init__(
        self,
        num_node_features,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = float(dropout)
        self.use_batchnorm = use_batchnorm

        # Build GCN layers
        self.convs = nn.ModuleList()
        in_channels = num_node_features
        for i in range(num_layers):
            out_channels = hidden_channels
            self.convs.append(GCNConv(in_channels, out_channels))
            in_channels = out_channels

        # Optional batchnorms for hidden dims
        if use_batchnorm and num_layers > 0:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        else:
            self.bns = None

        # Simple MLP readout: (mean+max pooled) -> hidden//2 -> 1
        # since we concatenate mean and max pool, first Linear expects 2*hidden_channels
        readout_hidden = max(hidden_channels // 2, 16)
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_channels, readout_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embedding through stacked GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply BatchNorm before the non-linearity (conv -> BN -> ReLU)
            if self.bns is not None:
                # BatchNorm1d expects shape [batch_size, features]; node dim works
                x = self.bns[i](x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout: concatenate mean and max pooled node features per graph
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        x = torch.cat([mean_pool, max_pool], dim=1)  # shape [batch_size, 2*hidden_channels]

        # MLP head -> returns shape [batch_size, 1]
        out = self.readout(x)
        return out
