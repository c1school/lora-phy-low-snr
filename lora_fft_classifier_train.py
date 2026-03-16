from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lora_baseline import LoRaParams, demodulate_dechirp_fft, generate_reference_upchirp


# ============================================================
# LoRa Dechirp+FFT Magnitude Classifier 학습 스크립트
# ------------------------------------------------------------
# 이 파일이 수행하는 작업:
# 1) npz 데이터셋(train/val/test) 로드
# 2) noisy 신호에 dechirp + FFT 수행
# 3) FFT magnitude를 입력으로 심볼 분류기 학습
# 4) 테스트셋에서 baseline vs classifier SER 비교
# 5) 모델 및 결과(csv, 그림) 저장
# ------------------------------------------------------------
# 필요 패키지:
#   pip install torch numpy matplotlib
# ============================================================


# ============================================================
# 설정값
# ============================================================


@dataclass
class TrainConfig:
    sf: int = 9
    bw: float = 125_000.0
    seed: int = 2026

    dataset_dir: str = os.path.join("dataset", "sf9_denoising")
    result_dir: str = os.path.join("results", "fft_classifier_sf9")

    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    num_workers: int = 0

    base_channels: int = 32

    @property
    def num_classes(self) -> int:
        return 2 ** self.sf


# ============================================================
# 유틸 함수
# ============================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: Tuple[torch.Tensor, ...], device: torch.device) -> Tuple[torch.Tensor, ...]:
    return tuple(x.to(device) for x in batch)


def twoch_to_complex_numpy(x_2ch: np.ndarray) -> np.ndarray:
    """
    (N, 2) 형태 [Re, Im] 배열을 복소수 (N,) 벡터로 변환
    """
    return x_2ch[:, 0].astype(np.float64) + 1j * x_2ch[:, 1].astype(np.float64)


def dechirp_and_fft_mag(noisy_2ch: np.ndarray, upchirp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    입력:
      noisy_2ch : (N, 2) = [Re, Im]
      upchirp   : (N,) complex

    출력:
      normalized magnitude spectrum, shape (N,)
    """
    noisy_complex = twoch_to_complex_numpy(noisy_2ch)
    dechirped = noisy_complex * np.conj(upchirp)
    spectrum = np.fft.fft(dechirped)
    mag = np.abs(spectrum)

    # log 압축 + 정규화
    mag = np.log1p(mag)
    mag = mag / (np.max(mag) + eps)

    return mag.astype(np.float32)


# ============================================================
# 데이터셋
# ============================================================


class LoRaFFTClassificationDataset(Dataset):
    def __init__(self, npz_path: str, params: LoRaParams) -> None:
        data = np.load(npz_path)
        self.X_noisy = data["X_noisy"]          # (num_samples, N, 2)
        self.symbol_index = data["symbol_index"]
        self.snr_db = data["snr_db"]

        self.upchirp = generate_reference_upchirp(params)

    def __len__(self) -> int:
        return len(self.symbol_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_2ch = self.X_noisy[idx]   # (N, 2)
        fft_mag = dechirp_and_fft_mag(noisy_2ch, self.upchirp)  # (N,)

        # Conv1d 입력 형태: (1, N)
        x = torch.from_numpy(fft_mag[None, :]).float()
        y = torch.tensor(int(self.symbol_index[idx]), dtype=torch.long)
        snr = torch.tensor(float(self.snr_db[idx]), dtype=torch.float32)

        return x, y, snr


# ============================================================
# 모델: FFT Magnitude Classifier
# ------------------------------------------------------------
# 입력: dechirp 후 FFT magnitude (B, 1, N)
# 출력: symbol logits (B, num_classes)
# ============================================================


class FFTMagnitudeClassifier1D(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 32) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # N -> N/2

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # N -> N/4

            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # N -> N/8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        logits = self.classifier(z)
        return logits


# ============================================================
# 학습 / 검증
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    loss_sum = 0.0
    correct_sum = 0
    total_count = 0

    for batch in loader:
        x, y, _ = to_device(batch, device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct_sum += int((torch.argmax(logits, dim=1) == y).sum().item())
        total_count += batch_size

    return {
        "loss": loss_sum / max(1, total_count),
        "acc": correct_sum / max(1, total_count),
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    correct_sum = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            x, y, _ = to_device(batch, device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            loss_sum += float(loss.item()) * batch_size
            correct_sum += int((torch.argmax(logits, dim=1) == y).sum().item())
            total_count += batch_size

    return {
        "loss": loss_sum / max(1, total_count),
        "acc": correct_sum / max(1, total_count),
    }


# ============================================================
# 평가: baseline vs classifier SER
# ============================================================


def evaluate_ser_by_snr(
    model: nn.Module,
    test_dataset: LoRaFFTClassificationDataset,
    params: LoRaParams,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    baseline:
      noisy -> dechirp + FFT -> pred symbol

    proposed:
      dechirp + FFT magnitude -> classifier -> pred symbol
    """
    model.eval()
    upchirp = generate_reference_upchirp(params)

    count_by_snr = defaultdict(int)
    base_err_by_snr = defaultdict(int)
    prop_err_by_snr = defaultdict(int)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            x_tensor, y_tensor, snr_tensor = test_dataset[idx]
            true_symbol = int(y_tensor.item())
            snr_db = float(snr_tensor.item())

            # baseline 계산을 위해 원래 noisy 샘플 재구성
            noisy_2ch = test_dataset.X_noisy[idx]
            noisy_complex = twoch_to_complex_numpy(noisy_2ch)
            pred_base, _, _ = demodulate_dechirp_fft(noisy_complex, upchirp)

            # proposed classifier
            model_in = x_tensor.unsqueeze(0).to(device)   # (1, 1, N)
            logits = model(model_in)
            pred_prop = int(torch.argmax(logits, dim=1).item())

            count_by_snr[snr_db] += 1
            base_err_by_snr[snr_db] += int(pred_base != true_symbol)
            prop_err_by_snr[snr_db] += int(pred_prop != true_symbol)

            if (idx + 1) % 1000 == 0:
                print(f"평가 진행 중... {idx + 1}/{len(test_dataset)}")

    snr_list = sorted(count_by_snr.keys(), reverse=True)
    baseline_ser = []
    proposed_ser = []
    sample_count = []

    for snr_db in snr_list:
        n = count_by_snr[snr_db]
        sample_count.append(n)
        baseline_ser.append(base_err_by_snr[snr_db] / n)
        proposed_ser.append(prop_err_by_snr[snr_db] / n)

    return {
        "snr_db": snr_list,
        "baseline_ser": baseline_ser,
        "proposed_ser": proposed_ser,
        "num_samples": sample_count,
    }


# ============================================================
# 결과 저장
# ============================================================


def save_train_csv(
    csv_path: str,
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i, (tr, va) in enumerate(zip(train_history, val_history), start=1):
            writer.writerow([i, tr["loss"], tr["acc"], va["loss"], va["acc"]])


def save_ser_csv(csv_path: str, ser_result: Dict[str, List[float]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", "baseline_ser", "proposed_ser", "num_samples"])
        for snr, b, p, n in zip(
            ser_result["snr_db"],
            ser_result["baseline_ser"],
            ser_result["proposed_ser"],
            ser_result["num_samples"],
        ):
            writer.writerow([snr, b, p, n])


def plot_loss_curve(
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
    save_path: str,
) -> None:
    train_loss = [x["loss"] for x in train_history]
    val_loss = [x["loss"] for x in val_history]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker="o", label="Train Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("FFT Magnitude Classifier Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_accuracy_curve(
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
    save_path: str,
) -> None:
    train_acc = [x["acc"] for x in train_history]
    val_acc = [x["acc"] for x in val_history]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_acc) + 1), train_acc, marker="o", label="Train Accuracy")
    plt.plot(range(1, len(val_acc) + 1), val_acc, marker="o", label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("FFT Magnitude Classifier Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_compare(ser_result: Dict[str, List[float]], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(ser_result["snr_db"], ser_result["baseline_ser"], marker="o", label="Baseline")
    plt.semilogy(ser_result["snr_db"], ser_result["proposed_ser"], marker="o", label="Proposed (FFT Classifier)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title("Baseline vs FFT Magnitude Classifier SER on Test Set")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 메인 실행부
# ============================================================


def main() -> None:
    cfg = TrainConfig()
    ensure_dir(cfg.result_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    params = LoRaParams(sf=cfg.sf, bw=cfg.bw, seed=cfg.seed)

    # ------------------------
    # 1) 데이터셋 로드
    # ------------------------
    train_path = os.path.join(cfg.dataset_dir, "train.npz")
    val_path = os.path.join(cfg.dataset_dir, "val.npz")
    test_path = os.path.join(cfg.dataset_dir, "test.npz")

    train_dataset = LoRaFFTClassificationDataset(train_path, params)
    val_dataset = LoRaFFTClassificationDataset(val_path, params)
    test_dataset = LoRaFFTClassificationDataset(test_path, params)

    print(f"학습 데이터 수: {len(train_dataset)}")
    print(f"검증 데이터 수: {len(val_dataset)}")
    print(f"테스트 데이터 수: {len(test_dataset)}")
    print(f"분류 클래스 수: {cfg.num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # ------------------------
    # 2) 모델 / loss / optimizer
    # ------------------------
    model = FFTMagnitudeClassifier1D(
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)

    # ------------------------
    # 3) 학습 루프
    # ------------------------
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    best_val_acc = -1.0
    best_model_path = os.path.join(cfg.result_dir, "best_fft_classifier.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        train_stat = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stat = validate_one_epoch(model, val_loader, criterion, device)

        train_history.append(train_stat)
        val_history.append(val_stat)

        print(
            f"[Epoch {epoch:02d}/{cfg.num_epochs:02d}] "
            f"train_loss={train_stat['loss']:.6f}, "
            f"train_acc={train_stat['acc']:.6f} | "
            f"val_loss={val_stat['loss']:.6f}, "
            f"val_acc={val_stat['acc']:.6f}"
        )

        if val_stat["acc"] > best_val_acc:
            best_val_acc = val_stat["acc"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 최고 성능 모델 저장: {best_model_path}")

    # 최고 성능 모델 다시 로드
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ------------------------
    # 4) 학습 곡선 저장
    # ------------------------
    save_train_csv(os.path.join(cfg.result_dir, "train_history.csv"), train_history, val_history)
    plot_loss_curve(train_history, val_history, os.path.join(cfg.result_dir, "loss_curve.png"))
    plot_accuracy_curve(train_history, val_history, os.path.join(cfg.result_dir, "accuracy_curve.png"))

    # ------------------------
    # 5) 테스트셋 SER 평가
    # ------------------------
    ser_result = evaluate_ser_by_snr(model, test_dataset, params, device)

    save_ser_csv(os.path.join(cfg.result_dir, "ser_compare.csv"), ser_result)
    plot_ser_compare(ser_result, os.path.join(cfg.result_dir, "ser_compare.png"))

    # ------------------------
    # 6) 요약 출력
    # ------------------------
    print("\n========== 결과 요약 ==========")
    print(f"최고 검증 정확도: {best_val_acc:.6f}")
    print(f"모델 저장 경로: {os.path.abspath(best_model_path)}")
    print(f"결과 폴더    : {os.path.abspath(cfg.result_dir)}")
    print("===============================")

    print("\nSNR별 SER 비교:")
    for snr, b, p in zip(ser_result["snr_db"], ser_result["baseline_ser"], ser_result["proposed_ser"]):
        print(f"SNR={snr:>5.1f} dB | baseline={b:.6f} | proposed={p:.6f}")


if __name__ == "__main__":
    main()