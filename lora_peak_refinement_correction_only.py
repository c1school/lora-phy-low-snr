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

from lora_baseline import LoRaParams, generate_reference_upchirp


# ============================================================
# Correction-Only Peak Refinement
# ------------------------------------------------------------
# 아이디어:
# 1) baseline peak가 이미 맞은 샘플은 학습에서 제외
# 2) baseline peak가 틀렸지만, 정답이 local window 안에 있는 샘플만 학습
# 3) 모델은 오직 "어느 방향으로 얼마나 고칠지"만 학습
# 4) 추론 시 confidence가 높을 때만 correction 적용
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
    result_dir: str = os.path.join("results", "peak_refinement_correction_only_sf9")

    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    num_workers: int = 0

    hidden_dim: int = 128
    window_radius: int = 8

    # validation에서 confidence threshold를 탐색할 후보들
    threshold_candidates: Tuple[float, ...] = (
        0.00, 0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70, 0.80, 0.85,
        0.90, 0.95, 0.99, 1.00
    )

    @property
    def window_size(self) -> int:
        return 2 * self.window_radius + 1

    @property
    def num_classes(self) -> int:
        # offset = -R..-1, +1..+R  => 총 2R개
        return 2 * self.window_radius


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

    mag = np.log1p(mag)
    mag = mag / (np.max(mag) + eps)
    return mag.astype(np.float32)


def circular_signed_offset(target_idx: int, center_idx: int, N: int) -> int:
    """
    center_idx 기준으로 target_idx가 몇 칸 옆에 있는지 signed offset 반환
    """
    diff = (target_idx - center_idx) % N
    if diff > N // 2:
        diff -= N
    return int(diff)


def circular_window(vec: np.ndarray, center_idx: int, radius: int) -> np.ndarray:
    """
    원형 인덱스를 고려해 center_idx 주변 [-radius, +radius] window 추출
    """
    N = len(vec)
    indices = [(center_idx + k) % N for k in range(-radius, radius + 1)]
    return vec[indices].astype(np.float32)


def offset_to_class(offset: int, radius: int) -> int:
    """
    offset in [-R..-1, +1..+R] 를 class index [0..2R-1] 로 변환
    """
    assert offset != 0
    assert abs(offset) <= radius

    if offset < 0:
        # -R -> 0, -1 -> R-1
        return offset + radius
    else:
        # +1 -> R, +R -> 2R-1
        return (offset - 1) + radius


def class_to_offset(cls: int, radius: int) -> int:
    """
    class index [0..2R-1] 를 offset [-R..-1, +1..+R] 로 변환
    """
    if cls < radius:
        return cls - radius
    else:
        return cls - radius + 1


# ============================================================
# 전처리
# ------------------------------------------------------------
# split 하나를 미리 분석해서 아래 정보를 저장
# - baseline peak
# - true symbol
# - snr
# - local window
# - offset
# - baseline이 틀렸는지
# - correction-only 학습 대상인지
# ============================================================


class PeakPreprocessed:
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

        self.X_window = np.zeros((num_samples, 1, window_size), dtype=np.float32)
        self.true_symbol = np.zeros((num_samples,), dtype=np.int64)
        self.baseline_peak = np.zeros((num_samples,), dtype=np.int64)
        self.snr_db = np.zeros((num_samples,), dtype=np.float32)
        self.offset = np.zeros((num_samples,), dtype=np.int64)

        self.is_baseline_error = np.zeros((num_samples,), dtype=np.bool_)
        self.is_local_correctable = np.zeros((num_samples,), dtype=np.bool_)

        baseline_error_count = 0
        local_correctable_count = 0

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
            self.offset[i] = offset

            baseline_error = (base_peak != true_idx)
            local_correctable = baseline_error and (1 <= abs(offset) <= window_radius)

            self.is_baseline_error[i] = baseline_error
            self.is_local_correctable[i] = local_correctable

            baseline_error_count += int(baseline_error)
            local_correctable_count += int(local_correctable)

        self.num_samples = num_samples
        self.baseline_error_ratio = baseline_error_count / num_samples
        self.local_correctable_ratio = local_correctable_count / num_samples


# ============================================================
# 학습용 데이터셋
# ------------------------------------------------------------
# correction-only:
#   baseline 오답 + local window 안에서 보정 가능한 샘플만 사용
# ============================================================


class CorrectionOnlyDataset(Dataset):
    def __init__(self, prep: PeakPreprocessed, window_radius: int) -> None:
        self.prep = prep
        self.window_radius = window_radius

        self.indices = np.where(prep.is_local_correctable)[0]
        self.X = prep.X_window[self.indices]
        self.true_symbol = prep.true_symbol[self.indices]
        self.baseline_peak = prep.baseline_peak[self.indices]
        self.snr_db = prep.snr_db[self.indices]
        self.offset = prep.offset[self.indices]

        self.y_label = np.array(
            [offset_to_class(int(off), window_radius) for off in self.offset],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx]).float()          # (1, W)
        y = torch.tensor(int(self.y_label[idx]), dtype=torch.long)
        return x, y


# ============================================================
# 모델
# ============================================================


class PeakRefinementNet(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
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
# class weight
# ============================================================


def compute_class_weights(dataset: CorrectionOnlyDataset, num_classes: int) -> torch.Tensor:
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
        x, y = to_device(batch, device)

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
            x, y = to_device(batch, device)

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
# threshold 탐색
# ------------------------------------------------------------
# full validation split에서 confidence threshold를 조정해서
# overall SER가 가장 낮은 threshold를 선택
# ============================================================


def evaluate_overall_ser_with_threshold(
    model: nn.Module,
    prep: PeakPreprocessed,
    window_radius: int,
    threshold: float,
    device: torch.device,
) -> float:
    model.eval()

    total_errors = 0
    total_count = prep.num_samples

    with torch.no_grad():
        for i in range(prep.num_samples):
            x = torch.from_numpy(prep.X_window[i]).unsqueeze(0).float().to(device)  # (1,1,W)
            true_symbol = int(prep.true_symbol[i])
            baseline_peak = int(prep.baseline_peak[i])

            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            conf, pred_cls = torch.max(prob, dim=1)

            conf = float(conf.item())
            pred_cls = int(pred_cls.item())

            if conf >= threshold:
                pred_offset = class_to_offset(pred_cls, window_radius)
                pred_prop = (baseline_peak + pred_offset) % prep.params.N
            else:
                pred_prop = baseline_peak

            total_errors += int(pred_prop != true_symbol)

    return total_errors / total_count


def search_best_threshold(
    model: nn.Module,
    val_prep: PeakPreprocessed,
    window_radius: int,
    threshold_candidates: Tuple[float, ...],
    device: torch.device,
) -> Tuple[float, List[Tuple[float, float]]]:
    results = []

    best_threshold = 0.00
    print(f"\n임시 고정 threshold 사용: {best_threshold:.2f}")
    best_ser = float("inf")

    for th in threshold_candidates:
        ser = evaluate_overall_ser_with_threshold(model, val_prep, window_radius, th, device)
        results.append((th, ser))
        print(f"[threshold search] th={th:.2f} -> val SER={ser:.6f}")

        if ser < best_ser:
            best_ser = ser
            best_threshold = th

    return best_threshold, results


# ============================================================
# 테스트셋 SER 평가
# ============================================================


def evaluate_ser_by_snr(
    model: nn.Module,
    prep: PeakPreprocessed,
    window_radius: int,
    threshold: float,
    device: torch.device,
) -> Dict[str, List[float] | int | float]:
    model.eval()

    count_by_snr = defaultdict(int)
    base_err_by_snr = defaultdict(int)
    prop_err_by_snr = defaultdict(int)

    total_samples = 0
    applied_count = 0
    baseline_error_count = 0
    baseline_correct_count = 0

    corrected_error_count = 0   # baseline 오답 -> proposed 정답
    corrupted_count = 0         # baseline 정답 -> proposed 오답
    changed_but_still_wrong = 0 # correction 적용했지만 여전히 틀림
    changed_and_still_correct = 0  # baseline도 정답, proposed도 정답인데 값은 같은 경우라 보통 0이어야 함

    conf_sum = 0.0
    conf_applied_sum = 0.0

    with torch.no_grad():
        for i in range(prep.num_samples):
            x = torch.from_numpy(prep.X_window[i]).unsqueeze(0).float().to(device)
            true_symbol = int(prep.true_symbol[i])
            baseline_peak = int(prep.baseline_peak[i])
            snr_db = float(prep.snr_db[i])

            pred_base = baseline_peak
            base_correct = (pred_base == true_symbol)

            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            conf, pred_cls = torch.max(prob, dim=1)

            conf = float(conf.item())
            pred_cls = int(pred_cls.item())
            conf_sum += conf

            # 기본값: baseline 유지
            pred_prop = baseline_peak
            applied = False

            if conf >= threshold:
                pred_offset = class_to_offset(pred_cls, window_radius)
                pred_prop = (baseline_peak + pred_offset) % prep.params.N
                applied = True
                applied_count += 1
                conf_applied_sum += conf

            prop_correct = (pred_prop == true_symbol)

            total_samples += 1
            count_by_snr[snr_db] += 1
            base_err_by_snr[snr_db] += int(not base_correct)
            prop_err_by_snr[snr_db] += int(not prop_correct)

            baseline_error_count += int(not base_correct)
            baseline_correct_count += int(base_correct)

            if not base_correct and prop_correct:
                corrected_error_count += 1

            if base_correct and not prop_correct:
                corrupted_count += 1

            if applied and (not base_correct) and (not prop_correct):
                changed_but_still_wrong += 1

            if applied and base_correct and prop_correct and pred_prop != pred_base:
                changed_and_still_correct += 1

            if (i + 1) % 1000 == 0:
                print(f"평가 진행 중... {i + 1}/{prep.num_samples}")

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

        # 진단용 요약
        "total_samples": total_samples,
        "applied_count": applied_count,
        "applied_ratio": applied_count / total_samples if total_samples > 0 else 0.0,
        "baseline_error_count": baseline_error_count,
        "baseline_correct_count": baseline_correct_count,
        "corrected_error_count": corrected_error_count,
        "corrupted_count": corrupted_count,
        "changed_but_still_wrong": changed_but_still_wrong,
        "changed_and_still_correct": changed_and_still_correct,
        "avg_conf_all": conf_sum / total_samples if total_samples > 0 else 0.0,
        "avg_conf_applied": conf_applied_sum / applied_count if applied_count > 0 else 0.0,
        "net_gain": corrected_error_count - corrupted_count,
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


def save_threshold_search_csv(csv_path: str, results: List[Tuple[float, float]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "val_ser"])
        for th, ser in results:
            writer.writerow([th, ser])


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
    plt.title("Correction-Only Peak Refinement Loss Curve")
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
    plt.title("Correction-Only Peak Refinement Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_compare(ser_result: Dict[str, List[float]], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(ser_result["snr_db"], ser_result["baseline_ser"], marker="o", label="Baseline")
    plt.semilogy(ser_result["snr_db"], ser_result["proposed_ser"], marker="o", label="Proposed (Correction-Only Peak Refinement)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title("Baseline vs Correction-Only Peak Refinement SER")
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
    # 1) full split 전처리
    # ------------------------
    train_prep = PeakPreprocessed(os.path.join(cfg.dataset_dir, "train.npz"), params, cfg.window_radius)
    val_prep = PeakPreprocessed(os.path.join(cfg.dataset_dir, "val.npz"), params, cfg.window_radius)
    test_prep = PeakPreprocessed(os.path.join(cfg.dataset_dir, "test.npz"), params, cfg.window_radius)

    print(f"train baseline error ratio      : {train_prep.baseline_error_ratio:.6f}")
    print(f"train local correctable ratio   : {train_prep.local_correctable_ratio:.6f}")
    print(f"val   baseline error ratio      : {val_prep.baseline_error_ratio:.6f}")
    print(f"val   local correctable ratio   : {val_prep.local_correctable_ratio:.6f}")
    print(f"test  baseline error ratio      : {test_prep.baseline_error_ratio:.6f}")
    print(f"test  local correctable ratio   : {test_prep.local_correctable_ratio:.6f}")

    # ------------------------
    # 2) correction-only 학습용 subset 구성
    # ------------------------
    train_dataset = CorrectionOnlyDataset(train_prep, cfg.window_radius)
    val_dataset = CorrectionOnlyDataset(val_prep, cfg.window_radius)

    print(f"correction-only train samples   : {len(train_dataset)}")
    print(f"correction-only val samples     : {len(val_dataset)}")
    print(f"window radius                   : {cfg.window_radius}")
    print(f"window size                     : {cfg.window_size}")
    print(f"num classes                     : {cfg.num_classes}")

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
    # 3) 모델 / loss / optimizer
    # ------------------------
    model = PeakRefinementNet(
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
    ).to(device)

    class_weights = compute_class_weights(train_dataset, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)

    # ------------------------
    # 4) 학습 루프
    # ------------------------
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    best_val_acc = -1.0
    best_model_path = os.path.join(cfg.result_dir, "best_peak_refinement_correction_only.pt")

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

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ------------------------
    # 5) validation에서 best threshold 탐색
    # ------------------------
    best_threshold, threshold_search_results = search_best_threshold(
        model=model,
        val_prep=val_prep,
        window_radius=cfg.window_radius,
        threshold_candidates=cfg.threshold_candidates,
        device=device,
    )

    print(f"\n선택된 best threshold: {best_threshold:.2f}")

    # ------------------------
    # 6) 테스트셋 SER 평가
    # ------------------------
    ser_result = evaluate_ser_by_snr(
        model=model,
        prep=test_prep,
        window_radius=cfg.window_radius,
        threshold=best_threshold,
        device=device,
    )

    # ------------------------
    # 7) 저장
    # ------------------------
    save_train_csv(os.path.join(cfg.result_dir, "train_history.csv"), train_history, val_history)
    save_threshold_search_csv(os.path.join(cfg.result_dir, "threshold_search.csv"), threshold_search_results)
    save_ser_csv(os.path.join(cfg.result_dir, "ser_compare.csv"), ser_result)

    plot_loss_curve(train_history, val_history, os.path.join(cfg.result_dir, "loss_curve.png"))
    plot_accuracy_curve(train_history, val_history, os.path.join(cfg.result_dir, "accuracy_curve.png"))
    plot_ser_compare(ser_result, os.path.join(cfg.result_dir, "ser_compare.png"))

    # ------------------------
    # 8) 요약 출력
    # ------------------------
    print("\n========== 결과 요약 ==========")
    print(f"최고 correction-only val accuracy: {best_val_acc:.6f}")
    print(f"선택된 threshold                : {best_threshold:.2f}")
    print(f"모델 저장 경로                 : {os.path.abspath(best_model_path)}")
    print(f"결과 폴더                      : {os.path.abspath(cfg.result_dir)}")
    print("================================")

    print("\nSNR별 SER 비교:")
    for snr, b, p in zip(ser_result["snr_db"], ser_result["baseline_ser"], ser_result["proposed_ser"]):
        print(f"SNR={snr:>5.1f} dB | baseline={b:.6f} | proposed={p:.6f}")

    print("\n========== 진단 정보 ==========")
    print(f"전체 샘플 수                 : {ser_result['total_samples']}")
    print(f"correction 적용 수           : {ser_result['applied_count']}")
    print(f"correction 적용 비율         : {ser_result['applied_ratio']:.6f}")
    print(f"baseline 정답 샘플 수        : {ser_result['baseline_correct_count']}")
    print(f"baseline 오답 샘플 수        : {ser_result['baseline_error_count']}")
    print(f"baseline 오답 -> 정답 교정 수 : {ser_result['corrected_error_count']}")
    print(f"baseline 정답 -> 오답 훼손 수 : {ser_result['corrupted_count']}")
    print(f"적용했지만 여전히 오답 수     : {ser_result['changed_but_still_wrong']}")
    print(f"평균 confidence(전체)        : {ser_result['avg_conf_all']:.6f}")
    print(f"평균 confidence(적용 샘플)   : {ser_result['avg_conf_applied']:.6f}")
    print(f"net gain (교정-훼손)         : {ser_result['net_gain']}")
    print("================================")

if __name__ == "__main__":
    main()