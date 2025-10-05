# depth-anything-comparative
**Depth-Anything V2**를 **모빌리티 환경**에 적용하기 위한 지식 증류, 프루닝, 양자화, TensorRT 기반 **최적화 전략** 비교 및 분석 프로젝트

> 목표: **정확도를 유지**하며 **모델 크기, 지연시간, 메모리**를 줄이는 효율적인 전략을 정리


<div align="center">
    <div style="display: flex;">
        <img src="https://github.com/user-attachments/assets/2d847590-608b-48ff-95a1-326d68e842ce" alt="모델 시각화" style="width: 800px;"/>
          <h3>경량화 모델 시각화</h3>
    <p>(Input, Base V2, FP32, Pruned(50%), Pruned(50%) + Dyn INT8, Dyn INT8)</p>
    </div>
</div>


---

## 🔎 개요 (Overview)
- 대상 모델: **Depth-Anything V2** (Vision Transformer 계열 단안 깊이 추정)  
- 데이터셋: **DDAD** (Dense Depth for Autonomous Driving), 일부 ETRI 자율주행 데이터 병행  
- 최적화 축:
  1) **지식 증류(KD)**: Depth-Anything V2 Base → MobileNetV2/축소형 학생  
  2) **직접 경량화**: **Pruning**, **Dynamic INT8 Quantization**(PyTorch)  
  3) **플랫폼 가속**: **ONNX → TensorRT** 변환, FP16 최적화 및 엔진 튜닝

---

## 🧪 주요 결과 (요약)
- **CPU (PyTorch)**  
  1) Dynamic INT8 적용 시 모델 크기 **−65.3%** (371.9MB → 129.0MB)
  2) 추론시간 **−10.4%** (1214ms → 1098.5ms)
- **GPU (TensorRT)**  
  1) FP16 + 해상도 560×560 + BS=1에서 지연시간 **33.78ms** 달성
  2) **약 61.5% 개선** (Base 87.85ms 대비) → **29.6 FPS**

> 정리: **CPU 환경**에서는 **Dynamic INT8**이, **GPU 환경**에서는 **FP16 TensorRT** 조합이 가장 효율적

---

## ⚙️ 환경 (실험 환경)
| 항목 | 버전 |
|---|---|
| PyTorch | 2.2.2 + CUDA 12.8 |
| ONNX / ONNX Runtime | 1.16.0 / 1.17.1 |
| TensorRT | 10.6.0 |
| xFormers | 0.0.30 |
| Segmentation Models PyTorch | 0.5.0 |

---

## 📊 대표 성능

| 세팅 | 정밀도 | 입력(H×W) | 배치 | 지연(ms) | FPS | 비고 |
|---|---:|---:|---:|---:|---:|---|
| PyTorch FP32 (Base) | FP32 | 560×560 | 1 | **87.85** | 11.4 | 기준 |
| TensorRT FP16 | FP16 | 560×560 | 1 | **33.78** | 29.6 | **−61.5%** 지연 |
| PyTorch Dyn INT8 | INT8(w) | 560×560 | 1 | **1098.5** | — | 크기 −65.3%, 지연 −10.4% |

---

## 🧩 설계 포인트 & 권장 전략
- **CPU 환경**: Dynamic INT8으로 모델 크기, 메모리 절감  
- **GPU 환경**: FP16 TensorRT로 실시간 성능 확보  
- **정확도 유지**: 학생 모델은 증강, Loss 설계 및 progressive unfreezing 적용

---

## 🧾 참고 문헌

- [Depth-Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [DDAD dataset](https://github.com/TRI-ML/DDAD)
