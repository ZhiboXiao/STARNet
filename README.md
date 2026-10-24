# StarNet

StarNet 是一个基于 PyTorch 的多传感器信号重建模型。它先通过按传感器分组的一维卷积提取序列特征，再使用带跳跃连接的 U 形编码器—解码器融合传感器与序列维度的信息，最后沿传感器维度输出更密集的重建结果。

本仓库提供模型、通用监督训练、验证和推理代码。数据路径、传感器数量、重建倍率及训练超参数由运行命令显式指定。

## 模型结构

```text
输入 [B, C, S, T]
  → 按传感器分组的 Conv1d：每个传感器独立提取序列特征
  → 重排为 [B, 64, S, T]，通过二维卷积与残差融合
  → 四级下采样编码器 + 四级上采样解码器 + 跳跃连接
  → 重建头 [B, R, S, T]
  → 按传感器与子位置交错排列
输出 [B, 1, S × R, T]
```

| 符号 | 含义 |
| --- | --- |
| `B` | 批大小 |
| `C` | 每个传感器的输入通道数 |
| `S` | 输入传感器数量 |
| `T` | 每个传感器的序列长度 |
| `R` | 沿传感器维度的整数重建倍率 |

网络使用 Tanh 激活和 BatchNorm。编码器特征宽度依次为 64、128、256、512、1024；选择双线性上采样时，最深层宽度为 512。默认解码器使用转置卷积，也可以通过 `--bilinear` 选择双线性插值。最终输出是连续值，没有 Sigmoid 或 Softmax 限制。

输出排列满足 `output[b, 0, s * R + r, t] = logits[b, r, s, t]`。训练标签必须使用相同顺序。`n_classes` 是原模型保留的参数名，在这里表示重建倍率 `R`。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `StarNet.py` | 模型及其卷积、下采样、上采样模块 |
| `dataset.py` | 按同名文件配对的 NumPy 数据接口 |
| `train.py` | 训练、可选验证、最佳检查点保存和断点续训 |
| `predict.py` | 根据检查点重建单个样本 |
| `requirements.txt` | 运行依赖 |
| `tests/test_starnet.py` | 模型、数据接口及训练到推理流程检查 |

## 安装

在 Python 环境中运行：

```bash
git clone https://github.com/ZhiboXiao/STARNet.git
cd STARNet
python -m pip install -r requirements.txt
```

使用 GPU 时，请安装与本机 CUDA 环境匹配的 PyTorch 构建。`--device auto` 会在 CUDA 可用时使用 CUDA，否则使用 CPU；也可以指定 `cpu` 或 `cuda:0` 等设备。Kornia 用于可选的 SSIM 损失。

## 准备数据

一个 `.npy` 文件存储一个样本，输入与标签通过**完全相同的文件名**配对：

```text
<train_inputs>/
    sample_001.npy
    sample_002.npy
<train_targets>/
    sample_001.npy
    sample_002.npy
```

- 输入形状：`[C, S, T]`；单通道时也可以保存为 `[S, T]`。
- 标签形状：`[1, S * R, T]`，也可以保存为 `[S * R, T]`。
- 数组应为有限实数，读取后转换为 `float32`。
- 默认 DataLoader 要求同一批次的样本形状一致。使用固定序列长度最直接；变长序列需要自行实现采样或 padding。
- 四次池化要求 `S >= 16` 且 `T >= 16`。训练时每个批次还需要满足 `B * floor(S / 16) * floor(T / 16) > 1`，包括最后一个批次，以保证 BatchNorm 可用。

数据加载器保留原始数值，不自动归一化、裁剪、插值或交换坐标轴。请在导出数组时统一输入与标签的数值尺度、传感器顺序和序列方向。验证数据使用独立目录，以相同规则组织。

如果已有自定义 Dataset，可让 `__getitem__` 返回 `(input_tensor, target_tensor)`，并在训练入口替换 `PairedArrayDataset`。模型和训练循环使用上述张量接口。

## 训练

以下命令中的 `<...>` 都需要替换为自己的路径或参数。命令写为一行，可在 PowerShell 或 Bash 中使用：

```text
python train.py --train-inputs "<train_inputs>" --train-targets "<train_targets>" --output-dir "<run_directory>" --sensors <S> --expansion <R> --in-channels <C> --epochs <total_epochs> --batch-size <batch_size> --lr <learning_rate> --device auto
```

可选参数：

| 参数 | 作用 |
| --- | --- |
| `--val-inputs`、`--val-targets` | 同时提供时，每轮结束后评估验证集 |
| `--loss l1` | 默认，平均绝对误差 |
| `--loss mse` | 均方误差 |
| `--loss l1-ssim` | `L1 + ssim_weight × SSIMLoss`；需同时提供 `--ssim-weight` 和 `--data-range` |
| `--ssim-window` | SSIM 窗口大小，默认 5，需为奇数并适合数据空间尺寸 |
| `--bilinear` | 使用双线性上采样 |
| `--weight-decay` | Adam 的权重衰减，默认 0 |
| `--workers` | 数据加载工作进程数，默认 0 |
| `--seed` | PyTorch 初始化与数据打乱的随机种子，默认 0 |
| `--resume` | 从已有训练检查点继续 |

优化器采用 Adam。`--data-range` 表示 SSIM 使用的动态范围，例如预先归一化到 `[0, 1]` 的数据使用 1；这个选项本身不执行归一化。选择适合任务的损失和数值尺度。

运行目录包含：

```text
<run_directory>/
    config.json       # 本次命令的参数
    metrics.jsonl     # 每轮训练/验证损失
    last.pt           # 最近一轮的完整检查点
    best.pt           # 最佳损失对应的完整检查点
```

有验证集时按验证损失选择 `best.pt`；没有验证集时按训练损失选择。每轮损失按样本数加权平均。新训练要求输出目录为空，以避免覆盖已有结果。

### 断点续训

在原训练命令后加入 `--resume "<run_directory>/last.pt"`，并将 `--epochs` 设置为**累计目标轮数**。例如检查点已完成 `E` 轮，要再训练 `K` 轮，就传入 `E + K`。

模型配置、损失配置和验证集路径需要与检查点保持一致。续训恢复模型、Adam 状态、已完成轮数、最佳损失以及 PyTorch/数据打乱的随机状态；学习率和权重衰减以检查点保存的优化器值为准。`config.json` 记录本次命令输入，检查点的 `optimizer` 字段记录实际优化器状态。不同设备或 CUDA 算法的数值结果可能不同。

## 推理

```text
python predict.py --checkpoint "<run_directory>/best.pt" --input "<input_sample.npy>" --output "<prediction.npy>" --device auto
```

脚本从检查点读取模型配置，输出形状为 `[1, S * R, T]` 的 NumPy 数组。输入预处理应与训练一致。输出文件需使用新的 `.npy` 文件名。

也可以直接在 Python 中调用模型：

```python
import torch
from StarNet import StarNet

# inputs 是已准备好的 float32 Tensor，形状为 [B, C, S, T]。
# expansion 为任务设定的正整数重建倍率。
model = StarNet(
    n_channels=inputs.shape[1],
    n_classes=expansion,
    sensors=inputs.shape[2],
).to(inputs.device)
model.eval()
with torch.inference_mode():
    outputs = model(inputs)
```

上述代码展示模型接口，模型参数为随机初始化。使用训练结果时，可按 `predict.py` 中的方式加载检查点。传感器数量会决定分组卷积参数形状，因此训练和推理必须使用相同的 `C`、`S`、`R` 及上采样方式；序列长度 `T` 可以在满足尺寸要求的前提下变化。

## 验证代码

```bash
python -m unittest discover -s tests -v
```

测试使用临时生成的合成数组，覆盖前向/反向传播、输出交错顺序、输入校验、数据配对、训练、验证、断点续训和推理。合成数据仅用于检查代码执行流程。
