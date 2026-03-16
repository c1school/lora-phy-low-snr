from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from lora_baseline import LoRaParams, generate_reference_upchirp


# ============================================================
# LoRa Top-K Candidate Analysis
# ------------------------------------------------------------
# 목적:
#   baseline dechirp + FFT에서 정답 심볼이 top-K peak 후보 안에
#   얼마나 자주 들어오는지 분석
#
# 확인할 핵심:
# 1) 전체 샘플 기준 top-K 포함률
# 2) baseline 오답 샘플 기준 top-K 포함률
# 3) SNR별 top-K 포함률
# 4) baseline 오답 샘플에서 정답 심볼의 rank 분포
# ============================================================


@dataclass
class AnalysisConfig:
    sf: int = 9
    bw: float = 125_000.0
    seed: int = 2026

    dataset_path: str = os.path.join("dataset", "sf9_denoising", "test.npz")
    result_dir: str = os.path.join("results", "topk_candidate_analysis_sf9")

    # 보고 싶은 K 후보들
    topk_list: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)

    # true rank 히스토그램을 어디까지 그릴지
    rank_plot_max: int = 50


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def twoch_to_complex_numpy(x_2ch: np.ndarray) -> np.ndarray:
    """
    (N, 2) 형태 [Re, Im] 배열을 복소수 (N,) 벡터로 변환
    """
    return x_2ch[:, 0].astype(np.float64) + 1j * x_2ch[:, 1].astype(np.float64)


def dechirp_and_fft_mag(noisy_2ch: np.ndarray, upchirp: np.ndarray) -> np.ndarray:
    """
    noisy signal -> dechirp -> FFT magnitude
    """
    noisy_complex = twoch_to_complex_numpy(noisy_2ch)
    dechirped = noisy_complex * np.conj(upchirp)
    spectrum = np.fft.fft(dechirped)
    mag = np.abs(spectrum)
    return mag.astype(np.float64)


def compute_true_rank(mag: np.ndarray, true_symbol: int) -> int:
    """
    FFT magnitude에서 true_symbol이 몇 번째로 큰지 rank를 계산
    rank=1 이면 가장 큰 peak
    """
    # true_symbol보다 큰 bin 개수 + 1
    return int(np.sum(mag > mag[true_symbol]) + 1)


def save_summary_csv(csv_path: str, rows: List[List[object]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group",
            "num_samples",
            "top1",
            "top2",
            "top3",
            "top5",
            "top10",
            "top20",
            "top50",
            "mean_true_rank",
            "median_true_rank",
        ])
        writer.writerows(rows)


def save_snr_csv(csv_path: str, rows: List[List[object]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr_db",
            "num_samples",
            "top1",
            "top2",
            "top3",
            "top5",
            "top10",
            "top20",
            "top50",
            "mean_true_rank",
            "median_true_rank",
        ])
        writer.writerows(rows)


def plot_topk_curve(topk_list: List[int], overall_rates: List[float], error_rates: List[float], save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(topk_list, overall_rates, marker="o", label="전체 샘플")
    plt.plot(topk_list, error_rates, marker="o", label="baseline 오답 샘플")
    plt.xlabel("K")
    plt.ylabel("Top-K 포함률")
    plt.title("정답 심볼의 Top-K Candidate 포함률")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_rank_histogram(ranks: List[int], rank_plot_max: int, save_path: str) -> None:
    if len(ranks) == 0:
        return

    clipped = [min(r, rank_plot_max) for r in ranks]
    bins = np.arange(1, rank_plot_max + 2)

    plt.figure(figsize=(9, 5))
    plt.hist(clipped, bins=bins, align="left", rwidth=0.9)
    plt.xlabel(f"True Symbol Rank (>{rank_plot_max} 는 {rank_plot_max}에 묶음)")
    plt.ylabel("샘플 수")
    plt.title("Baseline 오답 샘플에서 정답 심볼 Rank 분포")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    cfg = AnalysisConfig()
    ensure_dir(cfg.result_dir)

    params = LoRaParams(sf=cfg.sf, bw=cfg.bw, seed=cfg.seed)
    upchirp = generate_reference_upchirp(params)

    data = np.load(cfg.dataset_path)
    X_noisy = data["X_noisy"]          # (num_samples, N, 2)
    symbol_index = data["symbol_index"]
    snr_db = data["snr_db"]

    num_samples = len(symbol_index)
    topk_list = list(cfg.topk_list)

    # 전체 샘플 / baseline 오답 샘플 기준 통계
    overall_hit = {k: 0 for k in topk_list}
    error_hit = {k: 0 for k in topk_list}

    overall_ranks: List[int] = []
    error_ranks: List[int] = []

    baseline_error_count = 0

    # SNR별 통계
    snr_total = defaultdict(int)
    snr_hit = {k: defaultdict(int) for k in topk_list}
    snr_ranks = defaultdict(list)

    snr_error_total = defaultdict(int)
    snr_error_hit = {k: defaultdict(int) for k in topk_list}
    snr_error_ranks = defaultdict(list)

    for i in range(num_samples):
        noisy_2ch = X_noisy[i]
        true_symbol = int(symbol_index[i])
        snr = float(snr_db[i])

        mag = dechirp_and_fft_mag(noisy_2ch, upchirp)

        # baseline 예측 = top1
        pred_symbol = int(np.argmax(mag))
        true_rank = compute_true_rank(mag, true_symbol)

        overall_ranks.append(true_rank)
        snr_total[snr] += 1
        snr_ranks[snr].append(true_rank)

        for k in topk_list:
            if true_rank <= k:
                overall_hit[k] += 1
                snr_hit[k][snr] += 1

        if pred_symbol != true_symbol:
            baseline_error_count += 1
            error_ranks.append(true_rank)
            snr_error_total[snr] += 1
            snr_error_ranks[snr].append(true_rank)

            for k in topk_list:
                if true_rank <= k:
                    error_hit[k] += 1
                    snr_error_hit[k][snr] += 1

        if (i + 1) % 1000 == 0:
            print(f"분석 진행 중... {i + 1}/{num_samples}")

    # 전체 요약 출력
    print("\n========== Top-K Candidate Analysis ==========")
    print(f"전체 샘플 수           : {num_samples}")
    print(f"baseline 오답 샘플 수  : {baseline_error_count}")
    print(f"baseline 오답 비율     : {baseline_error_count / num_samples:.6f}")
    print("---------------------------------------------")

    overall_rates = []
    error_rates = []

    for k in topk_list:
        overall_rate = overall_hit[k] / num_samples if num_samples > 0 else 0.0
        error_rate = error_hit[k] / baseline_error_count if baseline_error_count > 0 else 0.0
        overall_rates.append(overall_rate)
        error_rates.append(error_rate)

        print(
            f"Top-{k:<2d} 포함률 | "
            f"전체={overall_rate:.6f} | "
            f"baseline 오답 샘플 기준={error_rate:.6f}"
        )

    print("---------------------------------------------")
    print(f"전체 샘플 평균 true rank      : {np.mean(overall_ranks):.4f}")
    print(f"전체 샘플 중앙값 true rank    : {np.median(overall_ranks):.4f}")
    if len(error_ranks) > 0:
        print(f"오답 샘플 평균 true rank      : {np.mean(error_ranks):.4f}")
        print(f"오답 샘플 중앙값 true rank    : {np.median(error_ranks):.4f}")
    print("=============================================\n")

    # summary csv
    summary_rows = [
        [
            "overall",
            num_samples,
            overall_hit.get(1, 0) / num_samples if num_samples > 0 else 0.0,
            overall_hit.get(2, 0) / num_samples if num_samples > 0 else 0.0,
            overall_hit.get(3, 0) / num_samples if num_samples > 0 else 0.0,
            overall_hit.get(5, 0) / num_samples if num_samples > 0 else 0.0,
            overall_hit.get(10, 0) / num_samples if num_samples > 0 else 0.0,
            overall_hit.get(20, 0) / num_samples if num_samples > 0 else 0.0,
            overall_hit.get(50, 0) / num_samples if num_samples > 0 else 0.0,
            float(np.mean(overall_ranks)),
            float(np.median(overall_ranks)),
        ],
        [
            "baseline_error_only",
            baseline_error_count,
            error_hit.get(1, 0) / baseline_error_count if baseline_error_count > 0 else 0.0,
            error_hit.get(2, 0) / baseline_error_count if baseline_error_count > 0 else 0.0,
            error_hit.get(3, 0) / baseline_error_count if baseline_error_count > 0 else 0.0,
            error_hit.get(5, 0) / baseline_error_count if baseline_error_count > 0 else 0.0,
            error_hit.get(10, 0) / baseline_error_count if baseline_error_count > 0 else 0.0,
            error_hit.get(20, 0) / baseline_error_count if baseline_error_count > 0 else 0.0,
            error_hit.get(50, 0) / baseline_error_count if baseline_error_count > 0 else 0.0,
            float(np.mean(error_ranks)) if len(error_ranks) > 0 else 0.0,
            float(np.median(error_ranks)) if len(error_ranks) > 0 else 0.0,
        ],
    ]
    save_summary_csv(os.path.join(cfg.result_dir, "topk_summary.csv"), summary_rows)

    # SNR별 csv (baseline 오답 샘플 기준)
    snr_rows = []
    for snr in sorted(snr_error_total.keys(), reverse=True):
        n = snr_error_total[snr]
        snr_rows.append([
            snr,
            n,
            snr_error_hit.get(1, {}).get(snr, 0) / n if n > 0 else 0.0,
            snr_error_hit.get(2, {}).get(snr, 0) / n if n > 0 else 0.0,
            snr_error_hit.get(3, {}).get(snr, 0) / n if n > 0 else 0.0,
            snr_error_hit.get(5, {}).get(snr, 0) / n if n > 0 else 0.0,
            snr_error_hit.get(10, {}).get(snr, 0) / n if n > 0 else 0.0,
            snr_error_hit.get(20, {}).get(snr, 0) / n if n > 0 else 0.0,
            snr_error_hit.get(50, {}).get(snr, 0) / n if n > 0 else 0.0,
            float(np.mean(snr_error_ranks[snr])) if len(snr_error_ranks[snr]) > 0 else 0.0,
            float(np.median(snr_error_ranks[snr])) if len(snr_error_ranks[snr]) > 0 else 0.0,
        ])
    save_snr_csv(os.path.join(cfg.result_dir, "topk_by_snr_error_only.csv"), snr_rows)

    # 그래프 저장
    plot_topk_curve(
        topk_list=topk_list,
        overall_rates=overall_rates,
        error_rates=error_rates,
        save_path=os.path.join(cfg.result_dir, "topk_recall_curve.png"),
    )

    plot_rank_histogram(
        ranks=error_ranks,
        rank_plot_max=cfg.rank_plot_max,
        save_path=os.path.join(cfg.result_dir, "true_rank_histogram_error_only.png"),
    )

    print(f"결과 저장 폴더: {os.path.abspath(cfg.result_dir)}")


if __name__ == "__main__":
    main()