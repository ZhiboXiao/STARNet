"""Behavior checks using temporary synthetic arrays, independent of any dataset."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from StarNet import StarNet
from dataset import PairedArrayDataset

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class StarNetTests(unittest.TestCase):
    def test_shapes_gradients_and_interleaving(self):
        for bilinear in (False, True):
            with self.subTest(bilinear=bilinear):
                model = StarNet(2, 3, bilinear=bilinear, sensors=17)
                captured = []
                hook = model.outc.register_forward_hook(lambda module, inputs, output: captured.append(output))
                output = model(torch.randn(2, 2, 17, 33))
                hook.remove()
                self.assertEqual(tuple(output.shape), (2, 1, 51, 33))
                for r in range(3):
                    torch.testing.assert_close(output[:, :, r::3], captured[0][:, r:r + 1])
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(model.inc.conv1[0].weight.grad).all())
                self.assertGreater(model.inc.conv1[0].weight.grad.abs().sum().item(), 0)

    def test_invalid_inputs(self):
        model = StarNet(1, 2, sensors=16)
        for shape in ((1, 2, 16, 32), (1, 1, 17, 32), (1, 1, 16, 8), (1, 1, 16, 16)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape))

    def test_pairing_and_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, targets = root / "inputs", root / "targets"
            inputs.mkdir()
            targets.mkdir()
            np.save(inputs / "a.npy", np.full((16, 32), 7, dtype=np.float32))
            np.save(targets / "a.npy", np.full((32, 32), 11, dtype=np.float32))
            data = PairedArrayDataset(inputs, targets, in_channels=1, sensors=16, expansion=2)
            x, y = data[0]
            self.assertEqual(tuple(x.shape), (1, 16, 32))
            self.assertEqual(x.max().item(), 7)
            self.assertEqual(y.max().item(), 11)
            np.save(inputs / "unpaired.npy", np.zeros((16, 32), dtype=np.float32))
            with self.assertRaises(ValueError):
                PairedArrayDataset(inputs, targets, in_channels=1, sensors=16, expansion=2)

    def test_train_resume_and_predict(self):
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

        def run(*args):
            result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            for name in ("inputs", "targets", "val_inputs", "val_targets"):
                (root / name).mkdir()
            for split in ("", "val_"):
                for index in range(2):
                    x = rng.random((16, 32), dtype=np.float32)
                    np.save(root / f"{split}inputs" / f"{index}.npy", x)
                    np.save(root / f"{split}targets" / f"{index}.npy", np.repeat(x, 2, axis=0))
            common = ["train.py", "--train-inputs", root / "inputs", "--train-targets", root / "targets",
                      "--val-inputs", root / "val_inputs", "--val-targets", root / "val_targets",
                      "--output-dir", root / "run", "--sensors", 16, "--expansion", 2,
                      "--batch-size", 2, "--lr", 0.0001, "--device", "cpu",
                      "--loss", "l1-ssim", "--ssim-weight", 0.1, "--data-range", 1]
            run(*common, "--epochs", 1)
            first = torch.load(root / "run/last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(first["epoch"], 1)
            before = first["model"]["outc.conv2.7.weight"].clone()
            del first
            run(*common, "--epochs", 2, "--resume", root / "run/last.pt")
            checkpoint = torch.load(root / "run/last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertFalse(torch.equal(before, checkpoint["model"]["outc.conv2.7.weight"]))
            self.assertTrue(checkpoint["optimizer"]["state"])
            records = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in records], [1, 2])
            self.assertTrue(all(np.isfinite(row["val_loss"]) for row in records))
            run("predict.py", "--checkpoint", root / "run/best.pt", "--input", root / "inputs/0.npy",
                "--output", root / "prediction.npy", "--device", "cpu")
            prediction = np.load(root / "prediction.npy")
            self.assertEqual(prediction.shape, (1, 32, 32))
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
