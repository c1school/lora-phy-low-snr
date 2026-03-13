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

# 핵심
# 기존(MSE기반): model(noisy) -> clean
# 변경(residual 디노이저): model(noisy) -> estimated_noise
# 최종 출력: denoised = noisy - estimated_noise
# 이 방식은 원래 신호를 통째로 새로 생성하지 않고, 잡음만 추정해서 빼는 구조라서 LoRa 같은 신호에서 더 자연스럽고 안정적일 가능성????일려나



# ============================================================
# LoRa Residual Denoiser 학습 스크립트
# ------------------------------------------------------------
# 이 파일이 수행하는 작업:
# 1) npz 데이터셋(train/val/test) 로드
# 2) 1D CNN 기반 residual denoiser 학습
# 3) 학습 곡선(loss) 저장
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
    result_dir: str = os.path.join("results", "residual_denoiser_sf9")

    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    num_workers: int = 0

    # CPU에서도 너무 무겁지 않게 설정
    hidden_channels: int = 32


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


def twoch_to_complex(x_2ch: np.ndarray) -> np.ndarray:
    """
    (N, 2) 형태 [Re, Im] 배열을 복소수 (N,) 벡터로 변환
    """
    return x_2ch[:, 0].astype(np.float64) + 1j * x_2ch[:, 1].astype(np.float64)


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
        # Conv1d 입력을 위해 (N, 2) -> (2, N) 으로 transpose
        x = torch.from_numpy(self.X_noisy[idx].T).float()
        y = torch.from_numpy(self.Y_clean[idx].T).float()
        m = torch.tensor(int(self.symbol_index[idx]), dtype=torch.long)
        snr = torch.tensor(float(self.snr_db[idx]), dtype=torch.float32)
        return x, y, m, snr


# ============================================================
# 모델: Residual Denoiser
# ------------------------------------------------------------
# 아이디어:
#   noisy -> 모델 -> estimated_noise
#   denoised = noisy - estimated_noise
# ============================================================


class ResidualDenoiser1D(nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()

        self.noise_estimator = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: noisy signal, shape = (B, 2, N)

        반환값:
          denoised         : 추정 clean signal
          estimated_noise  : 추정 noise
        """
        estimated_noise = self.noise_estimator(x)
        denoised = x - estimated_noise
        return denoised, estimated_noise


# ============================================================
# 학습 / 검증
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x, y, _, _ = to_device(batch, device)

        optimizer.zero_grad()

        denoised, estimated_noise = model(x)
        loss = criterion(denoised, y)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            x, y, _, _ = to_device(batch, device)

            denoised, estimated_noise = model(x)
            loss = criterion(denoised, y)

            batch_size = x.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(1, total_count)


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
    테스트 데이터셋에서 SNR별로 baseline / proposed SER를 계산한다.

    baseline:
      noisy -> dechirp + FFT -> pred symbol

    proposed:
      noisy -> residual denoiser -> dechirp + FFT -> pred symbol
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

            # baseline: noisy 그대로 복조
            noisy_2ch = x_tensor.permute(1, 0).cpu().numpy()   # (N, 2)
            noisy_complex = twoch_to_complex(noisy_2ch)
            pred_base, _, _ = demodulate_dechirp_fft(noisy_complex, upchirp)

            # proposed: residual denoiser 통과 후 복조
            model_in = x_tensor.unsqueeze(0).to(device)        # (1, 2, N)
            denoised_tensor, estimated_noise = model(model_in)
            denoised_2ch = denoised_tensor.squeeze(0).permute(1, 0).cpu().numpy()   # (N, 2)
            denoised_complex = twoch_to_complex(denoised_2ch)
            pred_prop, _, _ = demodulate_dechirp_fft(denoised_complex, upchirp)

            count_by_snr[snr_db] += 1
            base_err_by_snr[snr_db] += int(pred_base != true_symbol)
            prop_err_by_snr[snr_db] += int(pred_prop != true_symbol)

            if (idx + 1) % 1000 == 0:
                print(f"평가 진행 중... {idx + 1}/{len(test_dataset)}")

    # SNR을 0, -2, -4, ... 순으로 보기 좋게 정렬
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


def save_loss_csv(csv_path: str, train_losses: List[float], val_losses: List[float]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, tr, va])


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


def plot_loss_curve(train_losses: List[float], val_losses: List[float], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Residual Denoiser Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_compare(ser_result: Dict[str, List[float]], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(ser_result["snr_db"], ser_result["baseline_ser"], marker="o", label="Baseline")
    plt.semilogy(ser_result["snr_db"], ser_result["proposed_ser"], marker="o", label="Proposed (Residual Denoiser)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title("Baseline vs Residual Denoiser SER on Test Set")
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
    # 2) 모델 / 손실 / 옵티마이저
    # ------------------------
    model = ResidualDenoiser1D(hidden_channels=cfg.hidden_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)

    # ------------------------
    # 3) 학습 루프
    # ------------------------
    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val_loss = float("inf")
    best_model_path = os.path.join(cfg.result_dir, "best_residual_denoiser.pt")

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"[Epoch {epoch:02d}/{cfg.num_epochs:02d}] "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 최고 성능 모델 저장: {best_model_path}")

    # 최고 성능 모델 다시 로드
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ------------------------
    # 4) 학습 곡선 저장
    # ------------------------
    save_loss_csv(os.path.join(cfg.result_dir, "loss_history.csv"), train_losses, val_losses)
    plot_loss_curve(train_losses, val_losses, os.path.join(cfg.result_dir, "loss_curve.png"))

    # ------------------------
    # 5) 테스트셋 SER 평가
    # ------------------------
    params = LoRaParams(sf=cfg.sf, bw=cfg.bw, seed=cfg.seed)
    ser_result = evaluate_ser_by_snr(model, test_dataset, params, device)

    save_ser_csv(os.path.join(cfg.result_dir, "ser_compare.csv"), ser_result)
    plot_ser_compare(ser_result, os.path.join(cfg.result_dir, "ser_compare.png"))

    # ------------------------
    # 6) 요약 출력
    # ------------------------
    print("\n========== 결과 요약 ==========")
    print(f"최저 검증 손실: {best_val_loss:.6f}")
    print(f"모델 저장 경로: {os.path.abspath(best_model_path)}")
    print(f"결과 폴더    : {os.path.abspath(cfg.result_dir)}")
    print("===============================")

    print("\nSNR별 SER 비교:")
    for snr, b, p in zip(ser_result["snr_db"], ser_result["baseline_ser"], ser_result["proposed_ser"]):
        print(f"SNR={snr:>5.1f} dB | baseline={b:.6f} | proposed={p:.6f}")


if __name__ == "__main__":
    main()