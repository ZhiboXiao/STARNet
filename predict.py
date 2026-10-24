"""Run one NumPy sample through a trained StarNet checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch

from StarNet import StarNet
from dataset import load_array
from train import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output.suffix != ".npy":
        parser.error("output must end with .npy")
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = StarNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        output = model(load_array(args.input).unsqueeze(0).to(device))[0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    print(f"Saved {output.shape} to {args.output}")


if __name__ == "__main__":
    main()
