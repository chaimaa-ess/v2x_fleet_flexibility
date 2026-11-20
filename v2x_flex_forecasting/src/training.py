"""Shared PyTorch training utilities for EV fleet-flexibility forecasting."""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    gradient_clip_norm: Optional[float] = 1.0
    device: Optional[str] = None
    seed: int = 42
    verbose: bool = True


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    epochs_trained: int
    training_seconds: float
    history: Dict[str, list]
    stopped_early: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def _validate_loader(loader: DataLoader, name: str) -> None:
    if len(loader) == 0:
        raise ValueError(f"{name} DataLoader contains no batches.")


def _move_batch( batch: Mapping[str, torch.Tensor], device: torch.device,) -> Dict[str, torch.Tensor]:

    if "history" not in batch or "target" not in batch:
        raise KeyError('Each batch must contain "history" and "target".' )

    moved = {
        "history": batch["history"].to( device=device, dtype=torch.float32, non_blocking=True,),
        "target": batch["target"].to(   device=device, dtype=torch.float32,non_blocking=True,),
    }

    if "history_time" in batch:
        moved["history_time"] = batch["history_time"].to( device=device, dtype=torch.float32,non_blocking=True,)

    if "future_time" in batch:
        moved["future_time"] = batch[  "future_time"].to(device=device, dtype=torch.float32, non_blocking=True,)

    return moved

def _forward_model( model: nn.Module, batch: Mapping[str, torch.Tensor],) -> torch.Tensor:
    """
    Forward one batch through the model.
    Calendar covariates are passed only when they are present
    in the batch.
    """
    history = batch["history"]

    if "history_time" in batch or "future_time" in batch:
        return model(
           history=batch["history"],
            history_time=batch.get("history_time"),
            future_time=batch.get("future_time"),
        )

    return model(history)


def _validate_prediction_shape(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            "Model output and target shapes must match exactly: "
            f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}."
        )


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device,
                    gradient_clip_norm: Optional[float] = None) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        moved = _move_batch(batch, device)
        history = moved["history"]
        target = moved["target"]
        batch_size = history.shape[0]
        optimizer.zero_grad(set_to_none=True)
        prediction = _forward_model(  model,moved,)
        _validate_prediction_shape(prediction, target)
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}.")
        loss.backward()
        if gradient_clip_norm is not None:
            if gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be > 0 or None.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        moved = _move_batch(batch, device)
        history = moved["history"]
        target = moved["target"]
        prediction = _forward_model( model,moved,)
        _validate_prediction_shape(prediction, target)
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss: {loss.item()}.")
        total_loss += loss.item() * history.shape[0]
        total_samples += history.shape[0]
    return total_loss / total_samples


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Optional[TrainingConfig] = None,
                criterion: Optional[nn.Module] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                scheduler: Optional[Any] = None,
                checkpoint_path: Optional[str | Path] = None) -> TrainingResult:
    """Train with validation-only early stopping and restore the best state."""
    config = config or TrainingConfig()
    if config.max_epochs < 1 or config.patience < 1:
        raise ValueError("max_epochs and patience must be >= 1.")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.min_delta < 0:
        raise ValueError("Invalid optimizer/early-stopping configuration.")

    _validate_loader(train_loader, "Training")
    _validate_loader(val_loader, "Validation")
    set_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    criterion = criterion or nn.MSELoss()
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    log = {"epoch": [], "train_loss": [], "val_loss": [], "learning_rate": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    stopped_early = False
    start = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, config.gradient_clip_norm)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        log["epoch"].append(epoch)
        log["train_loss"].append(float(train_loss))
        log["val_loss"].append(float(val_loss))
        log["learning_rate"].append(float(lr))

        improved = val_loss < best_val - config.min_delta
        if improved:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val, config)
        else:
            no_improve += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if config.verbose:
            mark = " *" if improved else ""
            print(f"Epoch {epoch:03d}/{config.max_epochs:03d} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f} | lr={lr:.3e}{mark}")

        if no_improve >= config.patience:
            stopped_early = True
            if config.verbose:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                      f"best val={best_val:.6f}.")
            break

    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Training completed without a valid model state.")
    model.load_state_dict(best_state)

    return TrainingResult(best_epoch, best_val, len(log["epoch"]), elapsed,
                          log, stopped_early)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader,
            device: Optional[str | torch.device] = None,
            return_targets: bool = True) -> Dict[str, np.ndarray]:
    """Generate predictions in DataLoader order."""
    _validate_loader(loader, "Prediction")
    if device is None:
        try:
            resolved = next(model.parameters()).device
        except StopIteration:
            resolved = resolve_device(None)
    else:
        resolved = torch.device(device)
    model.to(resolved)
    model.eval()

    predictions, targets = [], []
    for batch in loader:
        moved = _move_batch(batch, resolved)
        target = moved["target"]
        prediction = _forward_model(model,moved,)
        _validate_prediction_shape(prediction, target)
        predictions.append(prediction.detach().cpu().numpy())
        if return_targets:
            targets.append(target.detach().cpu().numpy())

    result = {"predictions": np.concatenate(predictions, axis=0)}
    if return_targets:
        result["targets"] = np.concatenate(targets, axis=0)
    return result


def save_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int,
                    val_loss: float, config: TrainingConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_config": asdict(config),
    }, path)


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: Optional[str] = None) -> Dict[str, Any]:
    resolved = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=resolved, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "training_config": checkpoint.get("training_config"),
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
