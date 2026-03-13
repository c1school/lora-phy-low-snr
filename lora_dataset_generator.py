from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from lora_baseline import (
    LoRaParams,
    add_awgn,
    generate_reference_upchirp,
    generate_symbol,
)

# 얘가 하는일
# 1. lora_baseline.py를 이용해서
# 2. clean LoRa 심볼 생성
# 3. 거기에 AWGN 추가해서 noisy 심볼 생성
# 4. 이를 (N, 2) 형태의 Re/Im 2채널 데이터로 저장
# 5. train / val / test 세트를 npz로 저장
# 6. 저장되는 데이터는 이렇게 생깁니다.
# 7. X_noisy: 입력 데이터
# 8. Y_clean: 정답 데이터
# 9. symbol_index: 심볼 번호
# 10. snr_db: 각 샘플의 SNR

# ============================================================
# LoRa 데이터셋 생성기 (디노이징 / 딥러닝용)
# ------------------------------------------------------------
# 이 파일이 수행하는 작업:
#   입력(input)  : 잡음이 섞인 LoRa 유사 심볼
#   정답(target) : 깨끗한 LoRa 유사 심볼
#
# 저장 형식 (.npz):
#   X_noisy      : (num_samples, N, 2) float32
#   Y_clean      : (num_samples, N, 2) float32
#   symbol_index : (num_samples,) int32
#   snr_db       : (num_samples,) float32
#
# 채널 표현 방식:
#   마지막 차원 크기 2 = [실수부, 허수부]
# ============================================================


def ensure_dir(path: str) -> None:
    # 폴더가 없으면 생성
    os.makedirs(path, exist_ok=True)


def complex_to_2ch(x: np.ndarray) -> np.ndarray:
    """
    복소수 벡터 shape (N,) 을
    실수형 2채널 텐서 shape (N, 2) = [Re, Im] 형태로 변환한다.
    """
    return np.stack([np.real(x), np.imag(x)], axis=-1).astype(np.float32)


def generate_dataset(
    params: LoRaParams,
    num_samples: int,
    snr_db_choices: List[float],
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    noisy-clean 쌍으로 이루어진 심볼 데이터셋을 생성한다.

    각 샘플 생성 과정:
      1) 심볼 인덱스 m을 [0, 2^SF - 1] 범위에서 무작위 선택
      2) 깨끗한 복소 LoRa 유사 심볼 생성
      3) snr_db_choices 중 하나를 무작위 선택
      4) AWGN을 추가하여 noisy 신호 생성
      5) noisy/clean 신호를 [실수부, 허수부] 2채널 배열로 저장
    """
    rng = np.random.default_rng(seed)
    upchirp = generate_reference_upchirp(params)

    X_noisy = np.zeros((num_samples, params.N, 2), dtype=np.float32)
    Y_clean = np.zeros((num_samples, params.N, 2), dtype=np.float32)
    symbol_index = np.zeros((num_samples,), dtype=np.int32)
    snr_db_array = np.zeros((num_samples,), dtype=np.float32)

    for i in range(num_samples):
        m = int(rng.integers(0, params.M))
        snr_db = float(rng.choice(snr_db_choices))

        clean_symbol = generate_symbol(m, params, upchirp)
        noisy_symbol = add_awgn(clean_symbol, snr_db, rng)

        X_noisy[i] = complex_to_2ch(noisy_symbol)
        Y_clean[i] = complex_to_2ch(clean_symbol)
        symbol_index[i] = m
        snr_db_array[i] = snr_db

        if (i + 1) % max(1, num_samples // 10) == 0:
            print(f"생성 진행 중... {i + 1}/{num_samples}")

    return {
        "X_noisy": X_noisy,
        "Y_clean": Y_clean,
        "symbol_index": symbol_index,
        "snr_db": snr_db_array,
    }


def save_dataset_npz(
    save_path: str,
    dataset: Dict[str, np.ndarray],
) -> None:
    # 생성된 데이터셋을 압축된 npz 파일로 저장
    np.savez_compressed(
        save_path,
        X_noisy=dataset["X_noisy"],
        Y_clean=dataset["Y_clean"],
        symbol_index=dataset["symbol_index"],
        snr_db=dataset["snr_db"],
    )


def save_metadata_json(
    save_path: str,
    params: LoRaParams,
    num_samples: int,
    snr_db_choices: List[float],
    seed: int,
) -> None:
    # JSON 저장 전에 numpy 자료형을 파이썬 기본 자료형으로 변환. 내부 원소가 numpy.int64일 수 있기때문
    meta = {
        "params": {
            "sf": int(params.sf),
            "bw": float(params.bw),
            "fs": float(params.fs) if params.fs is not None else None,
            "seed": int(params.seed),
        },
        "M": int(params.M),
        "N": int(params.N),
        "symbol_duration_sec": float(params.symbol_duration),
        "num_samples": int(num_samples),
        "snr_db_choices": [float(x) for x in snr_db_choices],
        "seed": int(seed),
        "data_format": {
            "X_noisy": "(num_samples, N, 2), float32, 마지막 차원 = [real, imag]",
            "Y_clean": "(num_samples, N, 2), float32, 마지막 차원 = [real, imag]",
            "symbol_index": "(num_samples,), int32",
            "snr_db": "(num_samples,), float32",
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def make_split_datasets(
    params: LoRaParams,
    out_dir: str,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    train_snr_db_choices: List[float],
    test_snr_db_choices: List[float],
    base_seed: int = 2026,
) -> None:
    """
    train / val / test 분할 데이터셋을 생성한다.

    추천 사용 방식:
      - train/val은 일정한 SNR 범위에서 학습용으로 사용
      - test는 같거나 더 넓은 범위의 SNR에서 일반화 성능 평가
    """
    ensure_dir(out_dir)

    print("\n[1/3] 학습(train) 데이터 생성 중...")
    train_set = generate_dataset(
        params=params,
        num_samples=train_samples,
        snr_db_choices=train_snr_db_choices,
        seed=base_seed,
    )
    save_dataset_npz(os.path.join(out_dir, "train.npz"), train_set)
    save_metadata_json(
        os.path.join(out_dir, "train_meta.json"),
        params,
        train_samples,
        train_snr_db_choices,
        base_seed,
    )

    print("\n[2/3] 검증(validation) 데이터 생성 중...")
    val_set = generate_dataset(
        params=params,
        num_samples=val_samples,
        snr_db_choices=train_snr_db_choices,
        seed=base_seed + 1,
    )
    save_dataset_npz(os.path.join(out_dir, "val.npz"), val_set)
    save_metadata_json(
        os.path.join(out_dir, "val_meta.json"),
        params,
        val_samples,
        train_snr_db_choices,
        base_seed + 1,
    )

    print("\n[3/3] 테스트(test) 데이터 생성 중...")
    test_set = generate_dataset(
        params=params,
        num_samples=test_samples,
        snr_db_choices=test_snr_db_choices,
        seed=base_seed + 2,
    )
    save_dataset_npz(os.path.join(out_dir, "test.npz"), test_set)
    save_metadata_json(
        os.path.join(out_dir, "test_meta.json"),
        params,
        test_samples,
        test_snr_db_choices,
        base_seed + 2,
    )

    print("\n모든 데이터셋 저장이 완료되었습니다.")
    print(f"저장 폴더: {os.path.abspath(out_dir)}")


def inspect_npz(npz_path: str, num_preview: int = 3) -> None:
    """
    저장된 npz 파일의 shape, dtype, 일부 샘플 정보를 출력한다.
    """
    data = np.load(npz_path)
    print(f"\n확인 중: {npz_path}")
    for k in data.files:
        print(f"  {k:<12}: shape={data[k].shape}, dtype={data[k].dtype}")

    symbol_index = data["symbol_index"]
    snr_db = data["snr_db"]
    print("\n샘플 미리보기:")
    for i in range(min(num_preview, len(symbol_index))):
        print(f"  sample {i}: symbol={int(symbol_index[i])}, snr_db={float(snr_db[i]):.1f}")


def main() -> None:
    # --------------------------------------------------------
    # 첫 번째 디노이징 데이터셋 실험에서 바꿔볼 설정값
    # --------------------------------------------------------
    params = LoRaParams(sf=9, bw=125_000.0, seed=2026)

    # 추천 초기 설정:
    # - Train/Val: 중간~낮은 SNR 범위에서 학습
    # - Test: 더 넓은 범위(더 극단적인 값 포함)에서 성능 평가
    train_snr_db_choices = [float(x) for x in np.arange(-4, -25, -2)]
    test_snr_db_choices = [float(x) for x in np.arange(0, -31, -2)]
    # json.dump()가 파이썬 기본 자료형만 잘 저장하는데 리스트 안에 numpy.int64 값이 들어가 에러발생할수 있어서 형 변환.

    train_samples = 30000
    val_samples = 5000
    test_samples = 5000

    out_dir = os.path.join("dataset", f"sf{params.sf}_denoising")

    make_split_datasets(
        params=params,
        out_dir=out_dir,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_snr_db_choices=train_snr_db_choices,
        test_snr_db_choices=test_snr_db_choices,
        base_seed=params.seed,
    )

    # 저장된 데이터셋의 구조를 간단히 확인
    inspect_npz(os.path.join(out_dir, "train.npz"))
    inspect_npz(os.path.join(out_dir, "val.npz"))
    inspect_npz(os.path.join(out_dir, "test.npz"))


if __name__ == "__main__":
    main()