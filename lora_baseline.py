from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# LoRa 베이스라인 시뮬레이션
# ------------------------------------------------------------
# 이 파일이 수행하는 작업:
# 1) 단순화된 이산시간 LoRa 유사 chirp 심볼 생성
# 2) 목표 SNR에 맞춰 AWGN 추가
# 3) dechirp + FFT 방식으로 복조
# 4) SNR에 따른 SER(Symbol Error Rate) 측정
# 5) 그래프 저장
# ------------------------------------------------------------
# 필요 패키지:
#   pip install numpy matplotlib
# ============================================================


@dataclass
class LoRaParams:
    sf: int = 9                # 확산 지수(Spreading Factor)
    bw: float = 125_000.0      # 대역폭 [Hz] (보고서/설명용 메타데이터)
    fs: float | None = None    # 샘플링 주파수 [Hz], 기본값은 bw
    seed: int = 2026           # 재현 가능한 실험을 위한 시드값

    def __post_init__(self) -> None:
        if self.sf < 5 or self.sf > 12:
            raise ValueError("일반적인 LoRa 설정에서 sf는 [5, 12] 범위여야 합니다.")
        if self.fs is None:
            self.fs = self.bw

    @property
    def M(self) -> int:
        # LoRa 심볼 집합(alphabet)에 포함되는 전체 심볼 수
        return 2 ** self.sf

    @property
    def N(self) -> int:
        # 베이스라인 모델에서 한 심볼당 샘플 수
        # 여기서는 단순화를 위해 M과 동일하게 둠
        return self.M

    @property
    def symbol_duration(self) -> float:
        # 심볼 길이(초)
        return self.M / self.bw


# ============================================================
# 신호 생성부
# ============================================================


def generate_reference_upchirp(params: LoRaParams) -> np.ndarray:
    """
    단순화된 이산시간 LoRa 유사 upchirp를 생성한다.

    중요한 점:
    - 이 코드는 알고리즘 연구용으로 만든 깔끔한 베이스라인 모델이다.
    - dechirp 이후 각 심볼은 하나의 복소 tone이 되도록 설계되어 있어서,
      FFT peak 검출이 의도한 방식대로 동작한다.

    수식 설계 아이디어:
        up[n] = exp(j * pi * n^2 / N)
    그리고 심볼 m은 exp(j * 2*pi*m*n/N)를 곱해서 만들기 때문에,
    dechirp를 수행하면 FFT의 m번째 bin에 peak가 나타나는 순수 tone이 된다.
    """
    n = np.arange(params.N)
    upchirp = np.exp(1j * np.pi * (n ** 2) / params.N)
    return upchirp.astype(np.complex128)


def generate_symbol(symbol_index: int, params: LoRaParams, upchirp: np.ndarray | None = None) -> np.ndarray:
    """
    LoRa 유사 복소 baseband 심볼 1개를 생성한다.

    symbol[n] = upchirp[n] * exp(j * 2*pi*symbol_index*n/N)

    기준 upchirp의 켤레복소수(conj(upchirp))로 dechirp를 수행하면,
    symbol_index 위치에 FFT peak가 생기는 tone으로 바뀐다.
    """
    if not (0 <= symbol_index < params.M):
        raise ValueError(f"symbol_index는 [0, {params.M - 1}] 범위여야 합니다.")

    if upchirp is None:
        upchirp = generate_reference_upchirp(params)

    n = np.arange(params.N)
    tone = np.exp(1j * 2 * np.pi * symbol_index * n / params.N)
    symbol = upchirp * tone
    return symbol.astype(np.complex128)


# ============================================================
# 채널 모델
# ============================================================


def signal_power(x: np.ndarray) -> float:
    # 평균 신호 전력 계산
    return float(np.mean(np.abs(x) ** 2))


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    목표 SNR을 만족하도록 복소 AWGN을 신호에 추가한다.

    복소 잡음의 경우:
        noise = n_i + j*n_q
        각 성분(I, Q)의 분산은 noise_power / 2
    """
    p_signal = signal_power(x)
    snr_linear = 10 ** (snr_db / 10.0)
    p_noise = p_signal / snr_linear
    sigma = math.sqrt(p_noise / 2.0)

    noise = rng.normal(0.0, sigma, size=x.shape) + 1j * rng.normal(0.0, sigma, size=x.shape)
    return (x + noise).astype(np.complex128)


# ============================================================
# 베이스라인 복조기: dechirp + FFT
# ============================================================


def dechirp(rx: np.ndarray, upchirp: np.ndarray) -> np.ndarray:
    # 수신 신호에 기준 upchirp의 켤레복소수를 곱해 dechirp 수행
    return rx * np.conj(upchirp)


def fft_spectrum(x: np.ndarray) -> np.ndarray:
    # FFT를 수행해 주파수 영역 스펙트럼 계산
    return np.fft.fft(x)


def demodulate_dechirp_fft(rx: np.ndarray, upchirp: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    반환값:
      pred_symbol: 예측된 심볼 인덱스
      dechirped:   dechirp 이후의 시간영역 신호
      spectrum:    FFT 결과
    """
    dechirped = dechirp(rx, upchirp)
    spectrum = fft_spectrum(dechirped)
    pred_symbol = int(np.argmax(np.abs(spectrum)))
    return pred_symbol, dechirped, spectrum


# ============================================================
# 평가용 유틸리티
# ============================================================


def run_single_symbol_demo(symbol_index: int, snr_db: float, params: LoRaParams, rng: np.random.Generator) -> Dict[str, np.ndarray | int | float]:
    # 단일 심볼에 대해 송신 → 잡음 추가 → 복조 과정을 수행하고
    # 그래프 출력에 필요한 중간 결과를 모두 반환한다.
    upchirp = generate_reference_upchirp(params)
    tx = generate_symbol(symbol_index, params, upchirp)
    rx = add_awgn(tx, snr_db, rng)
    pred_symbol, dechirped, spectrum = demodulate_dechirp_fft(rx, upchirp)

    return {
        "symbol_index": symbol_index,
        "snr_db": snr_db,
        "tx": tx,
        "rx": rx,
        "dechirped": dechirped,
        "spectrum": spectrum,
        "pred_symbol": pred_symbol,
    }


def monte_carlo_ser(
    params: LoRaParams,
    snr_db_list: List[float],
    n_trials_per_snr: int = 2000,
) -> Dict[str, np.ndarray]:
    """
    베이스라인 dechirp + FFT 복조기에 대해
    Monte Carlo 방식으로 SER을 계산한다.
    """
    rng = np.random.default_rng(params.seed)
    upchirp = generate_reference_upchirp(params)

    ser_list = []
    total_errors_list = []

    for snr_db in snr_db_list:
        errors = 0
        for _ in range(n_trials_per_snr):
            symbol_index = int(rng.integers(0, params.M))
            tx = generate_symbol(symbol_index, params, upchirp)
            rx = add_awgn(tx, snr_db, rng)
            pred_symbol, _, _ = demodulate_dechirp_fft(rx, upchirp)
            errors += int(pred_symbol != symbol_index)

        ser = errors / n_trials_per_snr
        ser_list.append(ser)
        total_errors_list.append(errors)
        print(
            f"[SF={params.sf}] SNR={snr_db:>6.1f} dB | "
            f"SER={ser:.6f} | errors={errors}/{n_trials_per_snr}"
        )

    return {
        "snr_db": np.array(snr_db_list, dtype=float),
        "ser": np.array(ser_list, dtype=float),
        "errors": np.array(total_errors_list, dtype=int),
    }


def monte_carlo_ser_multi_sf(
    sf_list: List[int],
    snr_db_list: List[float],
    n_trials_per_snr: int = 2000,
    bw: float = 125_000.0,
) -> Dict[int, Dict[str, np.ndarray]]:
    # 여러 SF 값에 대해 SER 곡선을 각각 계산한다.
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for sf in sf_list:
        params = LoRaParams(sf=sf, bw=bw, seed=2026 + sf)
        results[sf] = monte_carlo_ser(params, snr_db_list, n_trials_per_snr=n_trials_per_snr)
    return results


# ============================================================
# 그래프 출력
# ============================================================


def ensure_dir(path: str) -> None:
    # 결과 저장용 폴더가 없으면 생성
    os.makedirs(path, exist_ok=True)


def plot_single_symbol_analysis(
    demo_result: Dict[str, np.ndarray | int | float],
    params: LoRaParams,
    save_path: str,
) -> None:
    # 단일 심볼에 대한 시간영역/Dechirp/FFT 결과를 한 그림에 저장한다.
    tx = np.asarray(demo_result["tx"])
    rx = np.asarray(demo_result["rx"])
    dechirped_signal = np.asarray(demo_result["dechirped"])
    spectrum = np.asarray(demo_result["spectrum"])
    true_symbol = int(demo_result["symbol_index"])
    pred_symbol = int(demo_result["pred_symbol"])
    snr_db = float(demo_result["snr_db"])

    n = np.arange(params.N)

    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(n[: min(128, params.N)], np.real(tx[: min(128, params.N)]), label="Re{tx}")
    plt.plot(n[: min(128, params.N)], np.real(rx[: min(128, params.N)]), label="Re{rx}", alpha=0.8)
    plt.title(f"Time-domain Symbol (first {min(128, params.N)} samples) | SF={params.sf}, SNR={snr_db} dB")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(n[: min(128, params.N)], np.real(dechirped_signal[: min(128, params.N)]), label="Re{dechirped}")
    plt.plot(n[: min(128, params.N)], np.imag(dechirped_signal[: min(128, params.N)]), label="Im{dechirped}", alpha=0.8)
    plt.title("Dechirped Signal (should look like a tone)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.abs(spectrum))
    plt.axvline(true_symbol, linestyle="--", label=f"True symbol={true_symbol}")
    plt.axvline(pred_symbol, linestyle=":", label=f"Pred symbol={pred_symbol}")
    plt.title("FFT Magnitude after Dechirp")
    plt.xlabel("FFT bin")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_curve(results: Dict[str, np.ndarray], params: LoRaParams, save_path: str) -> None:
    # 단일 SF에 대한 SER-SNR 곡선을 저장한다.
    snr_db = results["snr_db"]
    ser = results["ser"]

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr_db, ser, marker="o")
    plt.title(f"Baseline LoRa Demodulation SER vs SNR (SF={params.sf}, BW={params.bw/1000:.0f} kHz)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_ser_curve_multi_sf(results_by_sf: Dict[int, Dict[str, np.ndarray]], save_path: str) -> None:
    # 여러 SF 값에 대한 SER-SNR 곡선을 한 그림에 저장한다.
    plt.figure(figsize=(9, 6))
    for sf, result in results_by_sf.items():
        plt.semilogy(result["snr_db"], result["ser"], marker="o", label=f"SF={sf}")
    plt.title("Baseline LoRa Demodulation SER vs SNR for Multiple SF")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SER")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 메인 실험 실행부
# ============================================================


def main() -> None:
    # -----------------------------
    # 1) 가장 먼저 바꿔볼 수 있는 실험 설정값
    # -----------------------------
    single_sf = 9
    bw = 125_000.0
    n_trials_per_snr = 3000
    snr_db_list = list(np.arange(0, -31, -2))  # 0, -2, -4, ... -30
    multi_sf_list = [7, 8, 9, 10]

    # 결과 저장 폴더
    out_dir = "results"
    ensure_dir(out_dir)

    # --------------------------------
    # 2) 단일 심볼 분석용 그림 생성
    # --------------------------------
    params = LoRaParams(sf=single_sf, bw=bw, seed=2026)
    rng = np.random.default_rng(params.seed)

    demo_symbol = 123 % params.M
    demo_snr = -18.0
    demo_result = run_single_symbol_demo(demo_symbol, demo_snr, params, rng)

    plot_single_symbol_analysis(
        demo_result,
        params,
        save_path=os.path.join(out_dir, f"single_symbol_analysis_sf{single_sf}.png"),
    )

    print("\nSaved: single_symbol_analysis figure")
    print(
        f"True symbol = {demo_result['symbol_index']}, "
        f"Predicted symbol = {demo_result['pred_symbol']}"
    )

    # ------------------------
    # 3) 단일 SF에 대한 SER 곡선 생성
    # ------------------------
    print("\nRunning single-SF SER simulation...")
    single_sf_results = monte_carlo_ser(
        params=params,
        snr_db_list=snr_db_list,
        n_trials_per_snr=n_trials_per_snr,
    )
    plot_ser_curve(
        single_sf_results,
        params,
        save_path=os.path.join(out_dir, f"ser_curve_sf{single_sf}.png"),
    )
    print("Saved: single-SF SER curve")

    # -------------------------
    # 4) 여러 SF에 대한 SER 곡선 생성
    # -------------------------
    print("\nRunning multi-SF SER simulation...")
    multi_sf_results = monte_carlo_ser_multi_sf(
        sf_list=multi_sf_list,
        snr_db_list=snr_db_list,
        n_trials_per_snr=n_trials_per_snr,
        bw=bw,
    )
    plot_ser_curve_multi_sf(
        multi_sf_results,
        save_path=os.path.join(out_dir, "ser_curve_multi_sf.png"),
    )
    print("Saved: multi-SF SER curve")

    # -------------------------
    # 5) 실험 요약 출력
    # -------------------------
    print("\n========== Quick Summary ==========")
    print(f"BW                : {bw/1000:.0f} kHz")
    print(f"Single SF         : {single_sf}")
    print(f"Symbol duration   : {params.symbol_duration * 1e3:.3f} ms")
    print(f"Trials / SNR      : {n_trials_per_snr}")
    print(f"SNR points        : {snr_db_list}")
    print(f"Results directory : {os.path.abspath(out_dir)}")
    print("===================================")


if __name__ == "__main__":
    main()