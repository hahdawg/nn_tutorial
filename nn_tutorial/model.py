import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Classic feed forward network.

    Parameters
    ----------
    hidden_size: int
        Width of hidden layer
    num_outputs: int
        Number of classes to predict; for MNIST, it's 10
    device: str
        'cuda' or 'cpu'
    num_inputs (optional): int
        Number of input features; for MNIST, it's 28x28
    """
    def __init__(self, hidden_size: int, num_outputs: int, device: str, num_inputs: int = 0):
        # TODO: allow for multiple layers
        # TODO: try adding dropout, batchnorm, etc
        super().__init__()

        if num_inputs:
            self.hidden_layer = nn.Linear(in_features=num_inputs, out_features=hidden_size)
        else:
            self.hidden_layer = nn.LazyLinear(out_features=hidden_size)  # This is new
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=num_outputs)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: Tensor(size=(batch_size, num_features))

        Returns
        -------
        Tensor(size=(batch_size, num_outputs))
        """
        x = x.to(self.device)
        hidden = self.hidden_layer(x)
        hidden = self.relu(hidden)
        logits = self.output_layer(hidden)
        probs = self.softmax(logits)
        return probs


def test_ff():
    """
    Example of how to test a model
    """
    # Make dims as different/non-conformable as possible; will explain below
    hidden_size = 2
    batch_size = 3
    num_features = 5
    num_outputs = 7
    device = "cpu"
    model = FeedForward(hidden_size=hidden_size, num_outputs=num_outputs, device=device)
    x = torch.rand(size=(batch_size, num_features))
    p_hat = model(x)

    # We're generating probabilities so make sure 0 <= p_hat <= 1
    assert p_hat.min() > 0
    assert p_hat.max() < 1

    # Make sure output shape is correct; batch_size != num_outputs
    assert p_hat.shape == (batch_size, num_outputs)
