from collections import deque
from typing import Tuple

from numpy import mean
import torch
import torch.nn as nn
import torch.optim as optim

import nn_tutorial.model as ntm


def train_ff_model(
    hidden_size: int,
    num_steps: int,
    learning_rate: float,
    batch_generators: Tuple,
    logging_interval: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ntm.FeedForward(
        hidden_size=hidden_size,
        num_outputs=10,
        device=device
    )

    loss_fcn = nn.NLLLoss()  # TODO: make this a function param
    optimizer = optim.Adam(lr=learning_rate, params=model.parameters())  # TODO: make this a param

    # TODO: Evaluate bg_te after we're done training
    bg_tr, bg_val, _ = batch_generators

    running_loss_tr = deque(maxlen=logging_interval)
    running_loss_val = deque(maxlen=logging_interval)
    running_acc_val = deque(maxlen=logging_interval)

    # TODO: Implement early stopping
    # Hint: use torch.save whenever loss improves during the logging step
    for step, (batch_tr, batch_val) in enumerate(zip(bg_tr, bg_val)):
        if step >= num_steps:
            break

        # Ensure that we move training data to the right device
        X_tr, y_tr = batch_tr
        X_tr = X_tr.to(device)
        y_tr = y_tr.to(device)

        X_val, y_val = batch_val
        X_val = X_val.to(device)
        y_val = y_val.to(device)

        # Forward/Backward passs
        model.train()  # make sure model is in train mode
        optimizer.zero_grad()  # have to do this, because torch keeps sums of gradients by default
        p_hat = model(X_tr)  # forward pass
        loss = loss_fcn(p_hat, y_tr)
        loss.backward()  # take gradients
        optimizer.step()  # adjust parameters using "SGD step"

        running_loss_tr.append(loss.mean().item())

        # Evaluation step
        model.eval()  # put model in evaluation mode (e.g., for dropout)
        with torch.no_grad():  # don't take any gradients here
            p_hat = model(X_val)
            loss = loss_fcn(p_hat, y_val)
            running_loss_val.append(loss.mean().item())
            accuracy = (p_hat.argmax(dim=1) == y_val).float().mean()
            running_acc_val.append(accuracy.item())

        if step % logging_interval == 0:
            summary_tr = mean(running_loss_tr)
            summary_val_loss = mean(running_loss_val)
            summary_val_acc = mean(running_acc_val)
            msg = f"[step {step}]: loss_tr: {summary_tr:0.5f}    " + \
                f"loss_val: {summary_val_loss:0.5f}    acc_val: {summary_val_acc:0.5f}"
            print(msg)

    return model
