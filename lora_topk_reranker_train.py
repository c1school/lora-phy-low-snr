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
# LoRa Top-K Candidate Reranker
# ------------------------------------------------------------
# 아이디어:
# 1) baseline dechirp + FFT로 top-K 후보 추출
# 2) 각 후보 주변 local window + candidate feature 생성
# 3) shared network가 후보별 score를 계산
# 4) 가장 score 높은 후보를 선택
# 5) confidence threshold로 baseline 유지 여부 결정
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
    result_dir: str = os.path.join("results", "topk_reranker_sf9")

    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    num_workers: int = 0

    top_k: int = 20
    window_radius: int = 4
    candidate_hidden_dim: int = 64

    threshold_candidates: Tuple[float, ...] = (
        0.00, 0.05, 0.10, 0.15, 0.20,
        0.25, 0.30, 0.35, 0.40, 0.45,
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    )

    @property
    def window_size(self) -> int:
        return 2 * self.window_radius + 1


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
    (N, 2) = [Re, Im] -> complex (N,)
    """
    return x_2ch[:, 0].astype(np.float64) + 1j * x_2ch[:, 1].astype(np.float64)


def dechirp_and_fft_mag(noisy_2ch: np.ndarray, upchirp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    noisy signal -> dechirp -> FFT magnitude
    log 압축 + 정규화 포함
    """
    noisy_complex = twoch_to_complex_numpy(noisy_2ch)
    dechirped = noisy_complex * np.conj(upchirp)
    spectrum = np.fft.fft(dechirped)
    mag = np.abs(spectrum)

    mag = np.log1p(mag)
    mag = mag / (np.max(mag) + eps)
    return mag.astype(np.float32)


def circular_window(vec: np.ndarray, center_idx: int, radius: int) -> np.ndarray:
    """
    원형 인덱스를 고려해 center_idx 주변 [-radius, +radius] window 추출
    """
    N = len(vec)
    indices = [(center_idx + k) % N for k in range(-radius, radius + 1)]
    return vec[indices].astype(np.float32)


# ============================================================
# 전처리
# ------------------------------------------------------------
# full split을 미리 분석해서 아래를 저장
# - top-K candidate index
# - candidate별 local window
# - candidate peak value
# - candidate rank feature
# - baseline top1
# - true symbol
# - true symbol이 top-K 안에 있는지
# - hit sample이면 true position
# ============================================================


class TopKPreprocessed:
    def __init__(self, npz_path: str, params: LoRaParams, top_k: int, window_radius: int) -> None:
        data = np.load(npz_path)
        X_noisy = data["X_noisy"]          # (num_samples, N, 2)
        symbol_index = data["symbol_index"]
        snr_db = data["snr_db"]

        self.params = params
        self.top_k = top_k
        self.window_radius = window_radius
        self.upchirp = generate_reference_upchirp(params)

        num_samples = len(symbol_index)
        window_size = 2 * window_radius + 1

        self.candidate_windows = np.zeros((num_samples, top_k, 1, window_size), dtype=np.float32)
        self.candidate_peak = np.zeros((num_samples, top_k, 1), dtype=np.float32)
        self.candidate_rank = np.zeros((num_samples, top_k, 1), dtype=np.float32)
        self.candidate_index = np.zeros((num_samples, top_k), dtype=np.int64)

        self.true_symbol = np.zeros((num_samples,), dtype=np.int64)
        self.baseline_top1 = np.zeros((num_samples,), dtype=np.int64)
        self.snr_db = np.zeros((num_samples,), dtype=np.float32)
        self.true_pos_in_topk = np.full((num_samples,), fill_value=-1, dtype=np.int64)

        hit_count = 0
        baseline_error_count = 0
        baseline_error_and_hit_count = 0

        for i in range(num_samples):
            noisy_2ch = X_noisy[i]
            true_idx = int(symbol_index[i])
            snr = float(snr_db[i])

            mag = dechirp_and_fft_mag(noisy_2ch, self.upchirp)

            # top-K candidate 추출 (내림차순)
            topk_idx = np.argsort(mag)[::-1][:top_k]
            baseline_top1 = int(topk_idx[0])

            self.true_symbol[i] = true_idx
            self.baseline_top1[i] = baseline_top1
            self.snr_db[i] = snr
            self.candidate_index[i] = topk_idx

            for j, cand in enumerate(topk_idx):
                self.candidate_windows[i, j, 0, :] = circular_window(mag, int(cand), window_radius)
                self.candidate_peak[i, j, 0] = float(mag[int(cand)])
                self.candidate_rank[i, j, 0] = float(j) / max(1, top_k - 1)

            pos = np.where(topk_idx == true_idx)[0]
            if len(pos) > 0:
                self.true_pos_in_topk[i] = int(pos[0])
                hit_count += 1

            if baseline_top1 != true_idx:
                baseline_error_count += 1
                if len(pos) > 0:
                    baseline_error_and_hit_count += 1

        self.num_samples = num_samples
        self.topk_hit_ratio = hit_count / num_samples
        self.baseline_error_ratio = baseline_error_count / num_samples
        self.error_hit_ratio = (
            baseline_error_and_hit_count / baseline_error_count
            if baseline_error_count > 0 else 0.0
        )


# ============================================================
# 학습용 데이터셋
# ------------------------------------------------------------
# top-K 안에 true symbol이 존재하는 샘플만 사용
# label = true symbol의 top-K 내 위치
# ============================================================


class TopKHitDataset(Dataset):
    def __init__(self, prep: TopKPreprocessed) -> None:
        self.prep = prep
        self.indices = np.where(prep.true_pos_in_topk >= 0)[0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(self.indices[idx])

        x_window = torch.from_numpy(self.prep.candidate_windows[i]).float()   # (K, 1, W)
        x_peak = torch.from_numpy(self.prep.candidate_peak[i]).float()        # (K, 1)
        x_rank = torch.from_numpy(self.prep.candidate_rank[i]).float()        # (K, 1)
        y = torch.tensor(int(self.prep.true_pos_in_topk[i]), dtype=torch.long)
        return x_window, x_peak, x_rank, y


# ============================================================
# 모델
# ------------------------------------------------------------
# 입력:
#   candidate window  : (B, K, 1, W)
#   candidate peak    : (B, K, 1)
#   candidate rank    : (B, K, 1)
#
# 출력:
#   candidate score   : (B, K)
# ============================================================


class TopKRerankerNet(nn.Module):
    def __init__(self, window_size: int, candidate_hidden_dim: int = 64) -> None:
        super().__init__()

        self.window_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.candidate_mlp = nn.Sequential(
            nn.LazyLinear(candidate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(candidate_hidden_dim, 1),
        )

    def forward(
        self,
        x_window: torch.Tensor,   # (B, K, 1, W)
        x_peak: torch.Tensor,     # (B, K, 1)
        x_rank: torch.Tensor,     # (B, K, 1)
    ) -> torch.Tensor:
        B, K, _, W = x_window.shape

        # 후보별 shared encoding
        xw = x_window.reshape(B * K, 1, W)
        feat = self.window_encoder(xw)          # (B*K, F)
        feat = feat.reshape(B, K, -1)           # (B, K, F)

        cand_feat = torch.cat([feat, x_peak, x_rank], dim=-1)  # (B, K, F+2)

        scores = self.candidate_mlp(cand_feat).squeeze(-1)     # (B, K)
        return scores


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
        x_window, x_peak, x_rank, y = to_device(batch, device)

        optimizer.zero_grad()
        logits = model(x_window, x_peak, x_rank)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
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
            x_window, x_peak, x_rank, y = to_device(batch, device)

            logits = model(x_window, x_peak, x_rank)
            loss = criterion(logits, y)

            batch_size = y.size(0)
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
# full validation split에서 threshold를 조정하며
# overall SER가 가장 낮은 threshold를 선택
# ============================================================


def evaluate_overall_ser_with_threshold(
    model: nn.Module,
    prep: TopKPreprocessed,
    threshold: float,
    device: torch.device,
) -> float:
    model.eval()

    total_errors = 0

    with torch.no_grad():
        for i in range(prep.num_samples):
            x_window = torch.from_numpy(prep.candidate_windows[i]).unsqueeze(0).float().to(device)  # (1,K,1,W)
            x_peak = torch.from_numpy(prep.candidate_peak[i]).unsqueeze(0).float().to(device)        # (1,K,1)
            x_rank = torch.from_numpy(prep.candidate_rank[i]).unsqueeze(0).float().to(device)        # (1,K,1)

            true_symbol = int(prep.true_symbol[i])
            baseline_top1 = int(prep.baseline_top1[i])

            logits = model(x_window, x_peak, x_rank)
            prob = torch.softmax(logits, dim=1)
            conf, pred_pos = torch.max(prob, dim=1)

            conf = float(conf.item())
            pred_pos = int(pred_pos.item())

            if conf >= threshold:
                pred_symbol = int(prep.candidate_index[i, pred_pos])
            else:
                pred_symbol = baseline_top1

            total_errors += int(pred_symbol != true_symbol)

    return total_errors / prep.num_samples


def search_best_threshold(
    model: nn.Module,
    val_prep: TopKPreprocessed,
    threshold_candidates: Tuple[float, ...],
    device: torch.device,
) -> Tuple[float, List[Tuple[float, float]]]:
    results = []

    best_threshold = threshold_candidates[0]
    best_ser = float("inf")

    for th in threshold_candidates:
        ser = evaluate_overall_ser_with_threshold(model, val_prep, th, device)
        results.append((th, ser))
        print(f"[threshold search] th={th:.2f} -> val SER={ser:.6f}")

        if ser < best_ser:
            best_ser = ser
            best_threshold = th

    return best_threshold, results


# ============================================================
# 테스트셋 SER 평가 + 진단 정보
# ============================================================


def evaluate_ser_by_snr(
    model: nn.Module,
    prep: TopKPreprocessed,
    threshold: float,
    device: torch.device,
) -> Dict[str, List[float] | int | float]:
    model.eval()

    count_by_snr = defaultdict(int)
    base_err_by_snr = defaultdict(int)
    prop_err_by_snr = defaultdict(int)

    total_samples = 0
    applied_count = 0
    corrected_error_count = 0
    corrupted_count = 0
    baseline_error_count = 0
    baseline_correct_count = 0

    conf_sum = 0.0
    conf_applied_sum = 0.0

    with torch.no_grad():
        for i in range(prep.num_samples):
            x_window = torch.from_numpy(prep.candidate_windows[i]).unsqueeze(0).float().to(device)
            x_peak = torch.from_numpy(prep.candidate_peak[i]).unsqueeze(0).float().to(device)
            x_rank = torch.from_numpy(prep.candidate_rank[i]).unsqueeze(0).float().to(device)

            true_symbol = int(prep.true_symbol[i])
            baseline_top1 = int(prep.baseline_top1[i])
            snr_db = float(prep.snr_db[i])

            pred_base = baseline_top1
            base_correct = (pred_base == true_symbol)

            logits = model(x_window, x_peak, x_rank)
            prob = torch.softmax(logits, dim=1)
            conf, pred_pos = torch.max(prob, dim=1)

            conf = float(conf.item())
            pred_pos = int(pred_pos.item())
            conf_sum += conf

            pred_prop = baseline_top1
            if conf >= threshold:
                pred_prop = int(prep.candidate_index[i, pred_pos])
                applied_count += 1
                conf_applied_sum += conf

            prop_correct = (pred_prop == true_symbol)

            total_samples += 1
            count_by_snr[snr_db] += 1
            base_err_by_snr[snr_db] += int(not base_correct)
            prop_err_by_snr[snr_db] += int(not prop_correct)

            baseline_error_count += int(not base_correct)
            baseline_correct_count += int(base_correct)

            if (not base_correct) and prop_correct:
                corrected_error_count += 1

            if base_correct and (not prop_correct):
                corrupted_count += 1

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

        "total_samples": total_samples,
        "applied_count": applied_count,
        "applied_ratio": applied_count / total_samples if total_samples > 0 else 0.0,
        "baseline_error_count": baseline_error_count,
        "baseline_correct_count": baseline_correct_count,
        "corrected_error_count": corrected_error_count,
        "corrupted_count": corrupted_count,
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
    plt.title("Top-K Reranker Loss Curve")
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
    plt.title("Top-K Reranker Accuracy Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_compare(ser_result: Dict[str, List[float]], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(ser_result["snr_db"], ser_result["baseline_ser"], marker="o", label="Baseline")
    plt.semilogy(ser_result["snr_db"], ser_result["proposed_ser"], marker="o", label="Proposed (Top-K Reranker)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.title("Baseline vs Top-K Reranker SER")
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
    # 1) 전처리
    # ------------------------
    train_prep = TopKPreprocessed(os.path.join(cfg.dataset_dir, "train.npz"), params, cfg.top_k, cfg.window_radius)
    val_prep = TopKPreprocessed(os.path.join(cfg.dataset_dir, "val.npz"), params, cfg.top_k, cfg.window_radius)
    test_prep = TopKPreprocessed(os.path.join(cfg.dataset_dir, "test.npz"), params, cfg.top_k, cfg.window_radius)

    print(f"top_k                         : {cfg.top_k}")
    print(f"window_radius                 : {cfg.window_radius}")
    print(f"window_size                   : {cfg.window_size}")
    print(f"train top-K hit ratio         : {train_prep.topk_hit_ratio:.6f}")
    print(f"val   top-K hit ratio         : {val_prep.topk_hit_ratio:.6f}")
    print(f"test  top-K hit ratio         : {test_prep.topk_hit_ratio:.6f}")
    print(f"train baseline error ratio    : {train_prep.baseline_error_ratio:.6f}")
    print(f"val   baseline error ratio    : {val_prep.baseline_error_ratio:.6f}")
    print(f"test  baseline error ratio    : {test_prep.baseline_error_ratio:.6f}")
    print(f"train error-hit ratio         : {train_prep.error_hit_ratio:.6f}")
    print(f"val   error-hit ratio         : {val_prep.error_hit_ratio:.6f}")
    print(f"test  error-hit ratio         : {test_prep.error_hit_ratio:.6f}")

    # ------------------------
    # 2) 학습용 hit-only dataset
    # ------------------------
    train_dataset = TopKHitDataset(train_prep)
    val_dataset = TopKHitDataset(val_prep)

    print(f"train hit-only samples        : {len(train_dataset)}")
    print(f"val   hit-only samples        : {len(val_dataset)}")

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
    model = TopKRerankerNet(
        window_size=cfg.window_size,
        candidate_hidden_dim=cfg.candidate_hidden_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(model)

    # ------------------------
    # 4) 학습 루프
    # ------------------------
    train_history: List[Dict[str, float]] = []
    val_history: List[Dict[str, float]] = []

    best_val_acc = -1.0
    best_model_path = os.path.join(cfg.result_dir, "best_topk_reranker.pt")

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
    # 5) validation threshold 탐색
    # ------------------------
    best_threshold, threshold_search_results = search_best_threshold(
        model=model,
        val_prep=val_prep,
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
    print(f"최고 hit-only val accuracy : {best_val_acc:.6f}")
    print(f"선택된 threshold           : {best_threshold:.2f}")
    print(f"모델 저장 경로            : {os.path.abspath(best_model_path)}")
    print(f"결과 폴더                 : {os.path.abspath(cfg.result_dir)}")
    print("================================")

    print("\n========== 진단 정보 ==========")
    print(f"전체 샘플 수                 : {ser_result['total_samples']}")
    print(f"rerank 적용 수               : {ser_result['applied_count']}")
    print(f"rerank 적용 비율             : {ser_result['applied_ratio']:.6f}")
    print(f"baseline 정답 샘플 수        : {ser_result['baseline_correct_count']}")
    print(f"baseline 오답 샘플 수        : {ser_result['baseline_error_count']}")
    print(f"baseline 오답 -> 정답 교정 수 : {ser_result['corrected_error_count']}")
    print(f"baseline 정답 -> 오답 훼손 수 : {ser_result['corrupted_count']}")
    print(f"평균 confidence(전체)        : {ser_result['avg_conf_all']:.6f}")
    print(f"평균 confidence(적용 샘플)   : {ser_result['avg_conf_applied']:.6f}")
    print(f"net gain (교정-훼손)         : {ser_result['net_gain']}")
    print("================================")

    print("\nSNR별 SER 비교:")
    for snr, b, p in zip(ser_result["snr_db"], ser_result["baseline_ser"], ser_result["proposed_ser"]):
        print(f"SNR={snr:>5.1f} dB | baseline={b:.6f} | proposed={p:.6f}")


if __name__ == "__main__":
    main()