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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lora_baseline import LoRaParams, demodulate_dechirp_fft, generate_reference_upchirp


# ============================================================
# LoRa FFT-aware Denoiser 학습 스크립트
# ------------------------------------------------------------
# 이 파일이 수행하는 작업:
# 1) npz 데이터셋(train/val/test) 로드
# 2) 1D CNN 기반 denoiser 학습
# 3) 시간영역 MSE + dechirp 후 FFT magnitude loss를 함께 사용
# 4) 테스트셋에서 baseline vs proposed SER 비교
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
    result_dir: str = os.path.join("results", "fftaware_denoiser_sf9")

    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    num_workers: int = 0

    hidden_channels: int = 32

    # 전체 loss = alpha * time_loss + beta * fft_loss
    alpha_time: float = 1.0
    beta_fft: float = 0.5


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


def twoch_to_complex_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x shape: (B, 2, N)
    반환값: complex tensor shape (B, N)
    """
    return torch.complex(x[:, 0, :], x[:, 1, :])


# ============================================================
# 데이터셋
# ============================================================


class LoRaDenoiseDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        self.X_noisy = data["X_noisy"]          # (num_samples, N, 2)
        self.Y_clean = data["Y_clean"]          # (num_samples, N, 2)
        self.symbol_index = data["symbol_index"]
        self.snr_db = data["snr_db"]

    def __len__(self) -> int:
        return len(self.symbol_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Conv1d 입력을 위해 (N, 2) -> (2, N)
        x = torch.from_numpy(self.X_noisy[idx].T).float()
        y = torch.from_numpy(self.Y_clean[idx].T).float()
        m = torch.tensor(int(self.symbol_index[idx]), dtype=torch.long)
        snr = torch.tensor(float(self.snr_db[idx]), dtype=torch.float32)
        return x, y, m, snr


# ============================================================
# 모델: 1D CNN Denoiser
# ============================================================


class FFTAwareDenoiser1D(nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ============================================================
# FFT-aware Loss
# ------------------------------------------------------------
# total_loss = alpha * time_domain_MSE + beta * fft_domain_MSE
#
# fft_domain_MSE 계산 방식:
# 1) pred, target 각각 dechirp 수행
# 2) FFT magnitude 계산
# 3) 샘플별 최대값으로 정규화
# 4) 정규화된 magnitude MSE 계산
# ============================================================


class FFTAwareLoss(nn.Module):
    def __init__(
        self,
        upchirp_complex: torch.Tensor,
        alpha_time: float = 1.0,
        beta_fft: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha_time = alpha_time
        self.beta_fft = beta_fft
        self.eps = eps

        # 복소 upchirp를 buffer로 등록해서 device 이동 시 함께 따라가게 함
        self.register_buffer("upchirp_complex", upchirp_complex)

    def _normalized_fft_mag_after_dechirp(self, x_2ch: torch.Tensor) -> torch.Tensor:
        """
        x_2ch shape: (B, 2, N)
        반환값: 정규화된 FFT magnitude, shape (B, N)
        """
        x_complex = twoch_to_complex_torch(x_2ch)  # (B, N)

        # dechirp: x * conj(upchirp)
        dechirped = x_complex * torch.conj(self.upchirp_complex).unsqueeze(0)

        # FFT magnitude
        spectrum = torch.fft.fft(dechirped, dim=-1)
        mag = torch.abs(spectrum)

        # 샘플별 최대값으로 정규화
        mag = mag / (torch.amax(mag, dim=-1, keepdim=True) + self.eps)
        return mag

    def forward(self, pred_2ch: torch.Tensor, target_2ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_loss = F.mse_loss(pred_2ch, target_2ch)

        pred_mag = self._normalized_fft_mag_after_dechirp(pred_2ch)
        target_mag = self._normalized_fft_mag_after_dechirp(target_2ch)
        fft_loss = F.mse_loss(pred_mag, target_mag)

        total_loss = self.alpha_time * time_loss + self.beta_fft * fft_loss
        return total_loss, time_loss, fft_loss


# ============================================================
# 학습 / 검증
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: FFTAwareLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    total_loss_sum = 0.0
    time_loss_sum = 0.0
    fft_loss_sum = 0.0
    total_count = 0

    for batch in loader:
        x, y, _, _ = to_device(batch, device)

        optimizer.zero_grad()

        pred = model(x)
        total_loss, time_loss, fft_loss = criterion(pred, y)

        total_loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss_sum += float(total_loss.item()) * batch_size
        time_loss_sum += float(time_loss.item()) * batch_size
        fft_loss_sum += float(fft_loss.item()) * batch_size
        total_count += batch_size

    return {
        "total": total_loss_sum / max(1, total_count),
        "time": time_loss_sum / max(1, total_count),
        "fft": fft_loss_sum / max(1, total_count),
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: FFTAwareLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss_sum = 0.0
    time_loss_sum = 0.0
    fft_loss_sum = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            x, y, _, _ = to_device(batch, device)

            pred = model(x)
            total_loss, time_loss, fft_loss = criterion(pred, y)

            batch_size = x.size(0)
            total_loss_sum += float(total_loss.item()) * batch_size
            time_loss_sum += float(time_loss.item()) * batch_size
            fft_loss_sum += float(fft_loss.item()) * batch_size
            total_count += batch_size

    return {
        "total": total_loss_sum / max(1, total_count),
        "time": time_loss_sum / max(1, total_count),
        "fft": fft_loss_sum / max(1, total_count),
    }


# ============================================================
# 평가: baseline vs proposed SER
# ============================================================


def evaluate_ser_by_snr(
    model: nn.Module,
    test_dataset: LoRaDenoiseDataset,
    params: LoRaParams,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    baseline:
      noisy -> dechirp + FFT -> pred symbol

    proposed:
      noisy -> FFT-aware denoiser -> dechirp + FFT -> pred symbol
    """
    model.eval()
    upchirp = generate_reference_upchirp(params)

    count_by_snr = defaultdict(int)
    base_err_by_snr = defaultdict(int)
    prop_err_by_snr = defaultdict(int)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            x_tensor, _, m_tensor, snr_tensor = test_dataset[idx]
            true_symbol = int(m_tensor.item())
            snr_db = float(snr_tensor.item())

            # baseline
            noisy_2ch = x_tensor.permute(1, 0).cpu().numpy()   # (N, 2)
            noisy_complex = twoch_to_complex_numpy(noisy_2ch)
            pred_base, _, _ = demodulate_dechirp_fft(noisy_complex, upchirp)

            # proposed
            model_in = x_tensor.unsqueeze(0).to(device)        # (1, 2, N)
            denoised = model(model_in).squeeze(0).permute(1, 0).cpu().numpy()   # (N, 2)
            denoised_complex = twoch_to_complex_numpy(denoised)
            pred_prop, _, _ = demodulate_dechirp_fft(denoised_complex, upchirp)

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


def save_loss_csv(
    csv_path: str,
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total",
            "train_time",
            "train_fft",
            "val_total",
            "val_time",
            "val_fft",
        ])

        for i, (tr, va) in enumerate(zip(train_history, val_history), start=1):
            writer.writerow([
                i,
                tr["total"], tr["time"], tr["fft"],
                va["total"], va["time"], va["fft"],
            ])


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


def plot_total_loss_curve(
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
    save_path: str,
) -> None:
    train_total = [x["total"] for x in train_history]
    val_total = [x["total"] for x in val_history]

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_total) + 1), train_total, marker="o", label="Train Total Loss")
    plt.plot(range(1, len(val_total) + 1), val_total, marker="o", label="Val Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FFT-aware Denoiser Total Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_loss_components(
    train_history: List[Dict[str, float]],
    val_history: List[Dict[str, float]],
    save_path: str,
) -> None:
    epochs = range(1, len(train_history) + 1)

    train_time = [x["time"] for x in train_history]
    val_time = [x["time"] for x in val_history]
    train_fft = [x["fft"] for x in train_history]
    val_fft = [x["fft"] for x in val_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_time, marker="o", label="Train Time Loss")
    plt.plot(epochs, val_time, marker="o", label="Val Time Loss")
    plt.plot(epochs, train_fft, marker="o", label="Train FFT Loss")
    plt.plot(epochs, val_fft, marker="o", label="Val FFT Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("FFT-aware Denoiser Loss Components")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_compare(ser_result: Dict[str, List[float]], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(ser_result["snr_db"], ser_result["baseline_ser"], marker="o", label="Baseline")
    plt.semilogy(ser_result["snr_db"], ser_result["proposed_ser"], marker="o", label="Proposed (FFT-aware Denoiser)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title("Baseline vs FFT-aware Denoiser SER on Test Set")
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

    # ------------------------
    # 1) 데이터셋 로드
    # ------------------------
    train_path = os.path.join(cfg.dataset_dir, "train.npz")
    val_path = os.path.join(cfg.dataset_dir, "val.npz")
    test_path = os.path.join(cfg.dataset_dir, "test.npz")

    train_dataset = LoRaDenoiseDataset(train_path)
    val_dataset = LoRaDenoiseDataset(val_path)
    test_dataset = LoRaDenoiseDataset(test_path)

    print(f"학습 데이터 수: {len(train_dataset)}")
    print(f"검증 데이터 수: {len(val_dataset)}")
    print(f"테스트 데이터 수: {len(test_dataset)}")

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
    model = FFTAwareDenoiser1D(hidden_channels=cfg.hidden_channels).to(device)

    params = LoRaParams(sf=cfg.sf, bw=cfg.bw, seed=cfg.seed)
    upchirp = generate_reference_upchirp(params).astype(np.complex64)
    upchirp_torch = torch.from_numpy(upchirp).to(device)

    criterion = FFTAwareLoss(
        upchirp_complex=upchirp_torch,
        alpha_time=cfg.alpha_time,
        beta_fft=cfg.beta_fft,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)
    print(f"Loss 설정: total = {cfg.alpha_time} * time_loss + {cfg.beta_fft} * fft_loss")

    # ------------------------
    # 3) 학습 루프
    # ------------------------
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    best_val_total = float("inf")
    best_model_path = os.path.join(cfg.result_dir, "best_fftaware_denoiser.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        train_stat = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stat = validate_one_epoch(model, val_loader, criterion, device)

        train_history.append(train_stat)
        val_history.append(val_stat)

        print(
            f"[Epoch {epoch:02d}/{cfg.num_epochs:02d}] "
            f"train_total={train_stat['total']:.6f}, "
            f"train_time={train_stat['time']:.6f}, "
            f"train_fft={train_stat['fft']:.6f} | "
            f"val_total={val_stat['total']:.6f}, "
            f"val_time={val_stat['time']:.6f}, "
            f"val_fft={val_stat['fft']:.6f}"
        )

        if val_stat["total"] < best_val_total:
            best_val_total = val_stat["total"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 최고 성능 모델 저장: {best_model_path}")

    # 최고 성능 모델 다시 로드
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ------------------------
    # 4) 학습 곡선 저장
    # ------------------------
    save_loss_csv(os.path.join(cfg.result_dir, "loss_history.csv"), train_history, val_history)
    plot_total_loss_curve(train_history, val_history, os.path.join(cfg.result_dir, "loss_curve_total.png"))
    plot_loss_components(train_history, val_history, os.path.join(cfg.result_dir, "loss_curve_components.png"))

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
    print(f"최저 검증 Total Loss: {best_val_total:.6f}")
    print(f"모델 저장 경로: {os.path.abspath(best_model_path)}")
    print(f"결과 폴더    : {os.path.abspath(cfg.result_dir)}")
    print("===============================")

    print("\nSNR별 SER 비교:")
    for snr, b, p in zip(ser_result["snr_db"], ser_result["baseline_ser"], ser_result["proposed_ser"]):
        print(f"SNR={snr:>5.1f} dB | baseline={b:.6f} | proposed={p:.6f}")


if __name__ == "__main__":
    main()