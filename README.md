# LoRa PHY Baseline Simulation

Ultra-low SNR 환경에서 LoRa PHY Layer 신호의 복조 성능 개선 연구

## 지금까지 한거
- 단순화된 LoRa-like chirp symbol 생성
- AWGN 채널 추가
- dechirp + FFT 기반 baseline 복조
- SNR에 따른 SER 측정
- SF별 SER 비교
- 결과 그래프 저장

## 사용 환경
- Python(버전뭐쓰고있는지 기억안남)
- numpy
- matplotlib

## 실행 방법
```bash
pip install numpy matplotlib
python lora_baseline.py
```

또는

알아서 잘 ㄱㄱ