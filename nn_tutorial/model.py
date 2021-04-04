import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, hidden_size: int, num_outputs: int, device: str):
        # TODO: allow for multiple layers
        # TODO: try dropout, batchnorm, etc
        super().__init__()
        self.hidden_layer = nn.LazyLinear(out_features=hidden_size)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        hidden = self.hidden_layer(x)
        hidden = self.relu(hidden)
        logits = self.output_layer(hidden)
        probs = self.softmax(logits)
        return probs
