from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from lora_baseline import LoRaParams, generate_reference_upchirp


# ============================================================
# Tiny Overfit Test
# ------------------------------------------------------------
# 목적:
#   아주 작은 데이터셋을 모델이 거의 외울 수 있는지 확인
#
# 입력:
#   noisy signal -> dechirp -> FFT magnitude
#
# 출력:
#   symbol index classification
# ============================================================


@dataclass
class OverfitConfig:
    sf: int = 9
    bw: float = 125_000.0
    seed: int = 2026

    dataset_path: str = os.path.join("dataset", "sf9_denoising", "train.npz")
    result_dir: str = os.path.join("results", "tiny_overfit_test")

    num_classes: int = 512
    subset_size: int = 64
    batch_size: int = 16
    num_epochs: int = 200
    learning_rate: float = 1e-3


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def twoch_to_complex_numpy(x_2ch: np.ndarray) -> np.ndarray:
    return x_2ch[:, 0].astype(np.float64) + 1j * x_2ch[:, 1].astype(np.float64)


def dechirp_and_fft_mag(noisy_2ch: np.ndarray, upchirp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    noisy_complex = twoch_to_complex_numpy(noisy_2ch)
    dechirped = noisy_complex * np.conj(upchirp)
    spectrum = np.fft.fft(dechirped)
    mag = np.abs(spectrum)

    mag = np.log1p(mag)
    mag = mag / (np.max(mag) + eps)
    return mag.astype(np.float32)


class TinyFFTClassificationDataset(Dataset):
    def __init__(self, npz_path: str, params: LoRaParams) -> None:
        data = np.load(npz_path)
        self.X_noisy = data["X_noisy"]
        self.symbol_index = data["symbol_index"]
        self.upchirp = generate_reference_upchirp(params)

    def __len__(self) -> int:
        return len(self.symbol_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_2ch = self.X_noisy[idx]  # (N, 2)
        fft_mag = dechirp_and_fft_mag(noisy_2ch, self.upchirp)  # (N,)
        x = torch.from_numpy(fft_mag).float()                   # (N,)
        y = torch.tensor(int(self.symbol_index[idx]), dtype=torch.long)
        return x, y


class TinyMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    cfg = OverfitConfig()
    ensure_dir(cfg.result_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    params = LoRaParams(sf=cfg.sf, bw=cfg.bw, seed=cfg.seed)
    dataset = TinyFFTClassificationDataset(cfg.dataset_path, params)

    # 앞쪽 subset_size개만 사용
    subset_indices = list(range(cfg.subset_size))
    tiny_dataset = Subset(dataset, subset_indices)

    loader = DataLoader(
        tiny_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = TinyMLPClassifier(input_dim=2 ** cfg.sf, num_classes=cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)
    print(f"tiny subset size: {cfg.subset_size}")

    loss_history = []
    acc_history = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x, y in loader:
            x = x.to(device)   # (B, N)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((torch.argmax(logits, dim=1) == y).sum().item())
            total_count += batch_size

        avg_loss = total_loss / total_count
        avg_acc = total_correct / total_count

        loss_history.append(avg_loss)
        acc_history.append(avg_acc)

        if epoch == 1 or epoch % 10 == 0:
            print(f"[Epoch {epoch:03d}/{cfg.num_epochs}] loss={avg_loss:.6f}, acc={avg_acc:.6f}")

    # 마지막에 같은 tiny subset에서 다시 정확도 계산
    model.eval()
    final_correct = 0
    final_count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            final_correct += int((torch.argmax(logits, dim=1) == y).sum().item())
            final_count += x.size(0)

    final_acc = final_correct / final_count

    print("\n========== Tiny Overfit Test 결과 ==========")
    print(f"subset size : {cfg.subset_size}")
    print(f"final train accuracy on tiny subset : {final_acc:.6f}")
    print("===========================================")

    # 그래프 저장
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, cfg.num_epochs + 1), loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Tiny Overfit Test - Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "tiny_overfit_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, cfg.num_epochs + 1), acc_history, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Tiny Overfit Test - Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.result_dir, "tiny_overfit_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()