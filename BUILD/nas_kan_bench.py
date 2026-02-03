import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import pandas as pd
import numpy as np
import time

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from timm.utils import AverageMeter, accuracy
from kan_convolutional.KANLinear import KANLinear

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

KAN_SPACE = {
    "spline_order": [2, 3, 4, 5],
    "grid_size": [5, 8, 10, 15, 20],
    "base_activation": [nn.ReLU, nn.GELU, nn.SiLU],
}

CNN_SPACE = {
    "kernel_size": [1, 3, 5, 7],
    "padding": [0, 1],
}

class DynamicCNNKAN(nn.Module):
    def __init__(self, conv_cfgs, kan_cfg):
        super().__init__()
        layers = []
        in_ch = 3
        for cfg in conv_cfgs:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    cfg["out_ch"],
                    kernel_size=cfg["kernel_size"],
                    padding=cfg["padding"],
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_ch = cfg["out_ch"]
        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.kan = KANLinear(
            in_features=feat_dim,
            out_features=10,
            grid_size=kan_cfg["grid_size"],
            spline_order=kan_cfg["spline_order"],
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=kan_cfg["base_activation"],
            grid_eps=0.02,
            grid_range=[0, 1],
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.kan(x)
        return x

def generate_architectures(max_arch):
    architectures = []
    for num_conv in range(2, 8):
        for conv_params in itertools.product(
            CNN_SPACE["kernel_size"],
            CNN_SPACE["padding"],
            repeat=num_conv,
        ):
            kernels = conv_params[::2]
            paddings = conv_params[1::2]
            if any(kernels[i] > kernels[i + 1] for i in range(len(kernels) - 1)):
                continue
            h, w = 32, 32
            conv_cfgs = []
            out_ch = 8
            valid = True
            prev_out_ch = 0
            for k, p in zip(kernels, paddings):
                h = (h + 2 * p - k) // 1 + 1
                w = (w + 2 * p - k) // 1 + 1
                h = h // 2
                w = w // 2
                if h < 1 or w < 1:
                    valid = False
                    break
                if out_ch <= prev_out_ch:
                    valid = False
                    break
                conv_cfgs.append(
                    {
                        "out_ch": out_ch,
                        "kernel_size": k,
                        "padding": p,
                    }
                )
                prev_out_ch = out_ch
                out_ch *= 2
            if not valid:
                continue
            for s, g, a in itertools.product(
                KAN_SPACE["spline_order"],
                KAN_SPACE["grid_size"],
                KAN_SPACE["base_activation"],
            ):
                kan_cfg = {
                    "spline_order": s,
                    "grid_size": g,
                    "base_activation": a,
                }
                architectures.append((conv_cfgs, kan_cfg))
                if len(architectures) >= max_arch:
                    return architectures
    return architectures

def arch_to_string(conv_cfgs, kan_cfg):
    parts = []
    for c in conv_cfgs:
        parts.append(f"Conv2d(k={c['kernel_size']},p={c['padding']})")
    parts.append(
        f"KAN(order={kan_cfg['spline_order']},grid={kan_cfg['grid_size']},act={kan_cfg['base_activation'].__name__})"
    )
    return " -> ".join(parts)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def conv2d_flops(module, inp, out):
    x = inp[0]
    batch_size, Cin, H_in, W_in = x.shape
    Cout = module.out_channels
    Kh, Kw = module.kernel_size
    _, _, H_out, W_out = out.shape
    flops_per_sample = 2 * Cin * Cout * Kh * Kw * H_out * W_out
    return flops_per_sample

def kanlinear_flops(module, inp, out):
    x = inp[0]
    if x.dim() > 2:
        x = x.reshape(-1, module.in_features)
    d_in = module.in_features
    d_out = module.out_features
    K = module.spline_order
    G = module.grid_size
    flops_nl = d_in
    term = 9 * K * (G + 1.5 * K) + 2 * G - 2.5 * K + 3
    flops_main = (d_in * d_out) * term
    return flops_nl + flops_main

def linear_flops(module, inp, out):
    x = inp[0]
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    d_in = module.in_features
    d_out = module.out_features
    return 2 * d_in * d_out

def get_flops(model, input_res=(3, 32, 32), device=None):
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    flops_dict = {}
    def hook_conv(module, inp, out):
        flops_dict[module] = flops_dict.get(module, 0) + conv2d_flops(module, inp, out)
    def hook_kan(module, inp, out):
        flops_dict[module] = flops_dict.get(module, 0) + kanlinear_flops(module, inp, out)
    def hook_linear(module, inp, out):
        flops_dict[module] = flops_dict.get(module, 0) + linear_flops(module, inp, out)
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_conv))
        elif isinstance(m, KANLinear):
            hooks.append(m.register_forward_hook(hook_kan))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook_linear))
    dummy = torch.zeros(1, *input_res, device=device)
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    model.train(was_training)
    total_flops_per_batch1 = sum(flops_dict.values())
    return int(total_flops_per_batch1)

def estimate_flops(model):
    return get_flops(model, input_res=(3, 32, 32))

def train_eval_live(model, arch_id, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    epoch_train_loss = []
    epoch_train_acc = []
    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            acc1 = accuracy(out, y, topk=(1,))[0]
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc1.item(), x.size(0))
        epoch_train_loss.append(loss_meter.avg)
        epoch_train_acc.append(acc_meter.avg)
        print(
            f"[Arch {arch_id}] Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss_meter.avg:.4f} | "
            f"Acc: {acc_meter.avg:.2f}%"
        )
    total_time = time.time() - start_time
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    test_acc = accuracy_score(labels, preds)
    test_prec = precision_score(labels, preds, average="macro", zero_division=0)
    test_rec = recall_score(labels, preds, average="macro", zero_division=0)
    test_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return (
        test_acc,
        test_prec,
        test_rec,
        test_f1,
        total_time,
        epoch_train_loss,
        epoch_train_acc,
    )

architectures = generate_architectures(max_arch=100000000)

EPOCHS = 5

for i, (conv_cfgs, kan_cfg) in enumerate(architectures):
    print("\n" + "=" * 90)
    print(f"ARCHITECTURE {i+1}/{len(architectures)}")
    print(arch_to_string(conv_cfgs, kan_cfg))
    print("=" * 90)
    model = DynamicCNNKAN(conv_cfgs, kan_cfg)
    (
        acc,
        prec,
        rec,
        f1,
        epoch_time,
        epoch_train_loss,
        epoch_train_acc,
    ) = train_eval_live(model, i, epochs=EPOCHS)
    row = {
        "arch_id": i,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1_score": f1,
        "epoch_time_sec": epoch_time,
        "num_params": count_parameters(model),
        "num_flops": estimate_flops(model),
        "architecture": arch_to_string(conv_cfgs, kan_cfg),
    }
    for e in range(EPOCHS):
        row[f"train_loss_epoch_{e+1}"] = epoch_train_loss[e]
        row[f"train_acc_epoch_{e+1}"] = epoch_train_acc[e]
    df = pd.DataFrame([row])
    df.to_csv(
        "nas_cnn_kan_cifar10_results.csv",
        mode="a",
        index=False,
        header=(i == 0),
    )