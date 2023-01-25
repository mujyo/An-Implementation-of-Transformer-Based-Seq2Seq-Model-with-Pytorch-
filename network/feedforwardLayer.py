import torch
import torch.nn as nn

class PositionwiseFeedforwardLayer(nn.Module):
    """
    Transform the input from hidden_dim to pos_dim
    Args:
        hidden_dim: size of context vector
        posff_dim: size of positionwise feedforward layer
        dropout_p: dropout ratio

    Input: x
    - ** x ** -: context vector made from src

    Output: x
    - ** x ** -: x passed this layer
    """
    def __init__(self, hidden_dim, posff_dim, dropout_p):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, posff_dim)
        self.fc_2 = nn.Linear(posff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: (batch size, seq len, hidden dim) -> (batch size, seq len, hid dim)
        x = self.dropout(torch.relu(self.fc_1(x)))

        # x: (batch size, seq len, hid dim) -> (batch size, seq len, hid dim)
        x = self.fc_2(x)

        return x

