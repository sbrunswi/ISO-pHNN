"""Training utilities for PHNN models."""

import math
import torch
from torch import nn
from time import perf_counter

def train(model, train_loader, val_loader, epochs, output_weight=0.25, loss_fn="normalized_mse", device=None, weight_decay=0.01, validation_frequency=None, lr=1e-3):
    """
    Train a model with a simple supervised objective on (xdot, y).

    Parameters
    ----------
    model : torch.nn.Module
        Model that returns (xdot_hat, y_hat).
    train_loader : torch.utils.data.DataLoader
        Data loader yielding (X, u, xdot, y) batches.
    val_loader : torch.utils.data.DataLoader
        Data loader yielding (X, u, xdot, y) batches.
    epochs : int
        Number of training epochs.
    output_weight : float, optional
        Weight applied to the output loss term.
    loss_fn : str, optional
        Loss choice: "mse" or "normalized_mse".
    device : str or torch.device or None, optional
        Device for training. If None, uses the model's current device.
    weight decay : float, optional
        epresenting the AdamW weight decay
    validation_frequency:  int, optional
        how often to evaluate the model
    """
    if validation_frequency is None:
        validation_frequency = 10**100
    device = torch.device(device)
    model.to(device)
    print("Training device:", next(model.parameters()).device)

    # --- loss function (hard-coded options) ---
    if loss_fn == "mse":
        crit = nn.MSELoss()
    elif loss_fn == "normalized_mse":
        from utils import normalized_mse as crit  # keep it minimal, reuse same signature
    else:
        raise NotImplementedError("Loss function not implemented")
    from utils import normalized_mae as mae_fn
    # --- optimizer ---
    optimizer = torch.optim.AdamW(model.get_params(), lr=lr, weight_decay=weight_decay)

    # --- scheduler: cosine with 10% warmup, per-step ---
    total_steps = epochs * max(1, len(train_loader))
    warmup_frac = 0.1

    def lr_lambda(step):
        if total_steps == 0:
            return 1.0
        if step / total_steps < warmup_frac:
            return step / (total_steps * warmup_frac + 1e-12)
        actual_step = step - total_steps * warmup_frac
        actual_total = max(1, total_steps * (1 - warmup_frac))
        return 0.5 * (1.0 + math.cos(math.pi * actual_step / actual_total))


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    model.train()
    history = []
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
    train_start = perf_counter()
    for epoch in range(1, epochs + 1):
        running_xdot, running_y, running_total = 0.0, 0.0, 0.0
        for X, u, xdot, y in train_loader:
            X = X.float().to(device)
            u = u.float().to(device)
            xdot = xdot.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            xdot_hat, y_hat = model(X, u)

            loss_xdot = crit(xdot_hat, xdot)
            loss_y = crit(y_hat, y)
            loss = loss_xdot + output_weight * loss_y

            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            running_xdot += loss_xdot.item()
            running_y += loss_y.item()
            running_total += loss.item()

        n_batches = max(1, len(train_loader))

        epoch_xdot = running_xdot / n_batches
        epoch_y = running_y / n_batches
        epoch_total = running_total / n_batches
        current_lr = float(optimizer.param_groups[0]["lr"])

        record = {
            "epoch": int(epoch),
            "loss_xdot": float(epoch_xdot),
            "loss_y": float(epoch_y),
            "loss_total": float(epoch_total),
            "lr": current_lr,
            "val_mae_xdot": None,
        }

        log_parts = [
            f"Epoch {epoch}/{epochs}",
            f"loss_xdot: {epoch_xdot:.6f}",
            f"loss_y: {epoch_y:.6f}",
            f"loss_total: {epoch_total:.6f}",
            f"lr: {current_lr:.8e}",
        ]

        if epoch % validation_frequency == 0:
            if val_loader is not None and len(val_loader) > 0:
                xdot_cat = []
                xdot_hat_cat = []

                with torch.no_grad():
                    for X, u, xdot, y in val_loader:
                        X = X.float().to(device)
                        u = u.float().to(device)
                        xdot = xdot.float().to(device)
                        y = y.float().to(device)

                        xdot_hat, y_hat = model(X, u)

                        xdot_cat.append(xdot)
                        xdot_hat_cat.append(xdot_hat)

                xdot_cat = torch.cat(xdot_cat, 0)
                xdot_hat_cat = torch.cat(xdot_hat_cat, 0)

                val_mae = float(
                    mae_fn(xdot_hat_cat, xdot_cat)
                )

                record["val_mae_xdot"] = val_mae

                log_parts.append(
                    f"val_mae_xdot: {val_mae:.6f}"
                )

        history.append(record)

        print(" | ".join(log_parts))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
    training_seconds = perf_counter() - train_start

    model.train().cpu()

    return history, training_seconds