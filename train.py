"""Dataset-independent supervised training for StarNet."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from StarNet import StarNet
from dataset import PairedArrayDataset


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive.")
    return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-inputs", type=Path, required=True)
    p.add_argument("--train-targets", type=Path, required=True)
    p.add_argument("--val-inputs", type=Path)
    p.add_argument("--val-targets", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sensors", type=positive_int, required=True)
    p.add_argument("--expansion", type=positive_int, required=True)
    p.add_argument("--in-channels", type=positive_int, default=1)
    p.add_argument("--bilinear", action="store_true")
    p.add_argument("--epochs", type=positive_int, required=True, help="Total target epoch count, including resumed epochs")
    p.add_argument("--batch-size", type=positive_int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--loss", choices=["l1", "mse", "l1-ssim"], default="l1")
    p.add_argument("--ssim-weight", type=float)
    p.add_argument("--ssim-window", type=positive_int, default=5)
    p.add_argument("--data-range", type=float, help="Dynamic range for SSIM; use 1 for values normalized to [0, 1]")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if bool(args.val_inputs) != bool(args.val_targets):
        p.error("Supply both validation folders.")
    if args.sensors < 16 or args.workers < 0:
        p.error("sensors must be >= 16 and workers >= 0.")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("lr must be positive and weight-decay nonnegative; both must be finite.")
    if args.loss == "l1-ssim":
        if args.ssim_weight is None or not math.isfinite(args.ssim_weight) or args.ssim_weight < 0:
            p.error("l1-ssim requires a finite nonnegative --ssim-weight.")
        if args.data_range is None or not math.isfinite(args.data_range) or args.data_range <= 0:
            p.error("l1-ssim requires a finite positive --data-range.")
        if args.ssim_window % 2 != 1:
            p.error("ssim-window must be odd.")
    return args


def get_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def make_loss(args):
    if args.loss == "l1":
        return nn.L1Loss()
    if args.loss == "mse":
        return nn.MSELoss()
    from kornia.losses import SSIMLoss
    ssim = SSIMLoss(args.ssim_window, max_val=args.data_range)
    return lambda prediction, target: nn.functional.l1_loss(prediction, target) + args.ssim_weight * ssim(prediction, target)


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError("Prediction and target shapes differ.")
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; check data scaling and learning rate.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item() * inputs.shape[0]
            count += inputs.shape[0]
    return total / count


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    config = dict(n_channels=args.in_channels, n_classes=args.expansion, bilinear=args.bilinear, sensors=args.sensors)
    model = StarNet(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_loss(args)
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch, best = 0, float("inf")
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != config:
            raise ValueError("Resume model configuration differs from the checkpoint.")
        for key in ("loss", "ssim_weight", "ssim_window", "data_range", "val_inputs", "val_targets"):
            if checkpoint["train_config"][key] != run_config[key]:
                raise ValueError(f"Resume requires the same {key}; best loss must remain comparable.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best_loss"]
        generator.set_state(checkpoint["loader_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if args.epochs <= start_epoch:
            raise ValueError("epochs must exceed the completed checkpoint epoch.")
        print("Resumed optimizer settings:", optimizer.param_groups[0]["lr"], "learning rate", flush=True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("Use an empty output directory or --resume to avoid overwriting a run.")
    dataset_args = dict(in_channels=args.in_channels, sensors=args.sensors, expansion=args.expansion)
    train_data = PairedArrayDataset(args.train_inputs, args.train_targets, **dataset_args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=generator)
    val_loader = None
    if args.val_inputs:
        val_data = PairedArrayDataset(args.val_inputs, args.val_targets, **dataset_args)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device) if val_loader is not None else None
        score = val_loss if val_loss is not None else train_loss
        improved = score < best
        best = min(best, score)
        state = dict(epoch=epoch, model=model.state_dict(), model_config=config,
                     optimizer=optimizer.state_dict(), best_loss=best, train_config=run_config,
                     torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        save_checkpoint(args.output_dir / "last.pt", state)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", state)
        metrics = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, best_loss=best)
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
