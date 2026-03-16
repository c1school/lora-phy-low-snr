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
# LoRa Peak Refinement Training Script
# ------------------------------------------------------------
# 아이디어:
# 1) baseline으로 dechirp + FFT 수행
# 2) baseline peak 주변 작은 window를 잘라서 입력으로 사용
# 3) 신경망이 "몇 bin 옆이 진짜인지"를 예측
# 4) local 범위를 벗어나면 fallback class로 baseline 유지
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
    result_dir: str = os.path.join("results", "peak_refinement_sf9")

    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    num_workers: int = 0

    hidden_dim: int = 128

    # baseline peak 주변으로 볼 반경
    # 예: 8이면 [-8, ..., 0, ..., +8] 총 17개 bin
    window_radius: int = 8

    @property
    def window_size(self) -> int:
        return 2 * self.window_radius + 1

    @property
    def num_classes(self) -> int:
        # local offset class 수 + fallback class 1개
        # offset classes: [-R, ..., +R] -> 총 2R+1개
        # fallback: local window 밖이면 baseline 유지
        return self.window_size + 1


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
    noisy signal -> dechirp -> FFT magnitude
    """
    noisy_complex = twoch_to_complex_numpy(noisy_2ch)
    dechirped = noisy_complex * np.conj(upchirp)
    spectrum = np.fft.fft(dechirped)
    mag = np.abs(spectrum)

    # log 압축 + 정규화
    mag = np.log1p(mag)
    mag = mag / (np.max(mag) + eps)
    return mag.astype(np.float32)


def circular_signed_offset(target_idx: int, center_idx: int, N: int) -> int:
    """
    center_idx 기준으로 target_idx가 몇 칸 옆에 있는지
    원형(circular) 인덱스를 고려해서 signed offset으로 반환
    """
    diff = (target_idx - center_idx) % N
    if diff > N // 2:
        diff -= N
    return int(diff)


def circular_window(vec: np.ndarray, center_idx: int, radius: int) -> np.ndarray:
    """
    원형 인덱스를 고려해 center_idx 주변 [-radius, +radius] window를 추출
    """
    N = len(vec)
    indices = [(center_idx + k) % N for k in range(-radius, radius + 1)]
    return vec[indices].astype(np.float32)


# ============================================================
# 데이터셋
# ------------------------------------------------------------
# 미리 local window / label / baseline peak를 전부 전처리해서 저장
# ============================================================


class PeakRefinementDataset(Dataset):
    def __init__(self, npz_path: str, params: LoRaParams, window_radius: int) -> None:
        data = np.load(npz_path)
        X_noisy = data["X_noisy"]          # (num_samples, N, 2)
        symbol_index = data["symbol_index"]
        snr_db = data["snr_db"]

        self.params = params
        self.window_radius = window_radius
        self.upchirp = generate_reference_upchirp(params)

        num_samples = len(symbol_index)
        window_size = 2 * window_radius + 1
        fallback_class = window_size

        self.X_window = np.zeros((num_samples, 1, window_size), dtype=np.float32)
        self.y_label = np.zeros((num_samples,), dtype=np.int64)
        self.true_symbol = np.zeros((num_samples,), dtype=np.int64)
        self.baseline_peak = np.zeros((num_samples,), dtype=np.int64)
        self.snr_db = np.zeros((num_samples,), dtype=np.float32)
        self.is_local = np.zeros((num_samples,), dtype=np.bool_)

        local_count = 0

        for i in range(num_samples):
            noisy_2ch = X_noisy[i]
            true_idx = int(symbol_index[i])
            snr = float(snr_db[i])

            mag = dechirp_and_fft_mag(noisy_2ch, self.upchirp)
            base_peak = int(np.argmax(mag))

            offset = circular_signed_offset(true_idx, base_peak, self.params.N)

            self.X_window[i, 0, :] = circular_window(mag, base_peak, window_radius)
            self.true_symbol[i] = true_idx
            self.baseline_peak[i] = base_peak
            self.snr_db[i] = snr

            if abs(offset) <= window_radius:
                # offset class: [-R, ..., 0, ..., +R] -> [0, ..., 2R]
                label = offset + window_radius
                self.y_label[i] = label
                self.is_local[i] = True
                local_count += 1
            else:
                # local window 밖이면 fallback class
                self.y_label[i] = fallback_class
                self.is_local[i] = False

        self.local_ratio = local_count / num_samples

    def __len__(self) -> int:
        return len(self.y_label)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X_window[idx]).float()                 # (1, W)
        y = torch.tensor(int(self.y_label[idx]), dtype=torch.long)
        true_symbol = torch.tensor(int(self.true_symbol[idx]), dtype=torch.long)
        baseline_peak = torch.tensor(int(self.baseline_peak[idx]), dtype=torch.long)
        snr = torch.tensor(float(self.snr_db[idx]), dtype=torch.float32)
        return x, y, true_symbol, baseline_peak, snr


# ============================================================
# 모델
# ------------------------------------------------------------
# 입력: baseline peak 주변 local FFT magnitude window
# 출력:
#   -R ~ +R offset class
#   또는 fallback class
# ============================================================


class PeakRefinementNet(nn.Module):
    def __init__(self, window_size: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        logits = self.classifier(z)
        return logits


# ============================================================
# class weight 계산
# ------------------------------------------------------------
# fallback class가 많을 수 있으므로 약간 균형을 맞춰줌
# ============================================================


def compute_class_weights(dataset: PeakRefinementDataset, num_classes: int) -> torch.Tensor:
    counts = np.bincount(dataset.y_label, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


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
        x, y, _, _, _ = to_device(batch, device)

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
            x, y, _, _, _ = to_device(batch, device)

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
# 평가: baseline vs peak refinement
# ============================================================


def evaluate_ser_by_snr(
    model: nn.Module,
    test_dataset: PeakRefinementDataset,
    params: LoRaParams,
    window_radius: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    baseline:
      baseline peak 그대로 사용

    proposed:
      local window -> offset / fallback 예측
      - offset 예측이면 baseline peak를 보정
      - fallback 예측이면 baseline 유지
    """
    model.eval()

    count_by_snr = defaultdict(int)
    base_err_by_snr = defaultdict(int)
    prop_err_by_snr = defaultdict(int)

    fallback_class = 2 * window_radius + 1

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            x_tensor, _, true_symbol_tensor, baseline_peak_tensor, snr_tensor = test_dataset[idx]

            true_symbol = int(true_symbol_tensor.item())
            baseline_peak = int(baseline_peak_tensor.item())
            snr_db = float(snr_tensor.item())

            pred_base = baseline_peak

            model_in = x_tensor.unsqueeze(0).to(device)  # (1, 1, W)
            logits = model(model_in)
            pred_class = int(torch.argmax(logits, dim=1).item())

            if pred_class == fallback_class:
                pred_prop = baseline_peak
            else:
                pred_offset = pred_class - window_radius
                pred_prop = (baseline_peak + pred_offset) % params.N

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
    plt.title("Peak Refinement Loss Curve")
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
    plt.title("Peak Refinement Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_compare(ser_result: Dict[str, List[float]], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(ser_result["snr_db"], ser_result["baseline_ser"], marker="o", label="Baseline")
    plt.semilogy(ser_result["snr_db"], ser_result["proposed_ser"], marker="o", label="Proposed (Peak Refinement)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title("Baseline vs Peak Refinement SER on Test Set")
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

    train_dataset = PeakRefinementDataset(train_path, params, cfg.window_radius)
    val_dataset = PeakRefinementDataset(val_path, params, cfg.window_radius)
    test_dataset = PeakRefinementDataset(test_path, params, cfg.window_radius)

    print(f"학습 데이터 수: {len(train_dataset)}")
    print(f"검증 데이터 수: {len(val_dataset)}")
    print(f"테스트 데이터 수: {len(test_dataset)}")
    print(f"window radius : {cfg.window_radius}")
    print(f"window size   : {cfg.window_size}")
    print(f"num classes   : {cfg.num_classes}")
    print(f"train local ratio: {train_dataset.local_ratio:.4f}")
    print(f"val   local ratio: {val_dataset.local_ratio:.4f}")
    print(f"test  local ratio: {test_dataset.local_ratio:.4f}")

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
    model = PeakRefinementNet(
        window_size=cfg.window_size,
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    class_weights = compute_class_weights(train_dataset, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)

    # ------------------------
    # 3) 학습 루프
    # ------------------------
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    best_val_acc = -1.0
    best_model_path = os.path.join(cfg.result_dir, "best_peak_refinement.pt")

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
    ser_result = evaluate_ser_by_snr(model, test_dataset, params, cfg.window_radius, device)

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