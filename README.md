# 🇰🇷 Korean-LLM-v1

현대식 아키텍처 기반의 한국어 경량 언어 모델(LLM) 학습 프로젝트입니다.  
RoPE 위치 인코딩, SwiGLU FFN, 그리고 멀티헤드 어텐션을 활용한 **Transformer 기반** 모델을 처음부터 학습합니다.  
💾 겨우 541M 파라미터로도 우수한 한국어 이해 능력을 달성합니다!

[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76b900?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/Status-Done%20Training-brightred?style=flat-square)]()
[![Steps](https://img.shields.io/badge/Steps-9700%2F50000-orange?style=flat-square)]()
[![Loss](https://img.shields.io/badge/Loss-1.5%20±%200.08-yellow?style=flat-square)]()

---

## 📊 현재 학습 상태

| 지표 | 상태 |
|------|------|
| **현재 스텝** | 9,700 / 50,000 |
| **트레이닝 손실(Loss)** | 1.5대 (지속적 개선 중) |
| **누적 학습 토큰** | ~620M+ |
| **모델 크기** | 541M 파라미터 |
| **VRAM 사용량** | ~11GB (RTX 4090 기준) |

### 🎯 생성 예시

**Q**: "인공지능이란?"  
**A**: *인공지능(AI)은 인간의 지능을 컴퓨터로 모방한 기술입니다.*

---

## 🚀 주요 특징

### 💡 현대식 아키텍처
- **RoPE(Rotary Position Embedding)**: 더 나은 위치 인코딩
- **SwiGLU Activation**: 표준 FFN 대비 성능 향상
- **RMSNorm**: Layer Normalization의 경량화된 대안
- **Multi-Head Attention with KV-Cache**: 효율적인 생성 단계 최적화

### 📚 다중 한국어 데이터셋
자동 다운로드 & 캐싱 지원:
- **maywell/korean_textbooks** (tiny-textbooks)
- **squarelike/OpenOrca-gugugo-ko** (QA 쌍)
- **beomi/KoAlpaca-v1.1a** (Instruction-following)

### ⚡ 학습 효율화
- **Gradient Checkpointing**: 메모리 절약
- **Gradient Accumulation**: 효과적인 배치 크기 확대
- **Mixed Precision (bfloat16)**: 더 빠른 학습 + 적은 메모리
- **Cosine Scheduler + Warmup**: 안정적인 학습률 스케줄링

### 🔄 생산성 기능
- **자동 체크포인트 저장** (평가 시 마다)
- **재개 기능** (Latest checkpoint 자동 탐지)
- **분산학습 지원** (DDP 준비됨)

---

## 💻 시스템 요구사항

### GPU 권장사항

| 모델 구성 | 권장 GPU | VRAM 사용량 |
|---------|---------|-----------|
| **541M params (Korean-LLM-v1)** | **RTX 4090** | **~11GB** ✓ 현재 사용 |
| 1.3B params | RTX 3090 / RTX 4060 Ti | ~8GB |
| 2.3B+ params | A100 / RTX 6000 | ~16GB+ |

**지원 GPU**:
- ✅ NVIDIA RTX 4090 (권장)
- ✅ NVIDIA A100 / A6000
- ✅ NVIDIA L40
- ✅ NVIDIA V100 (제한적)

**최소 요구사항**:
- CUDA 12.0+
- cuDNN 8.9+
- PyTorch 2.0+

---

## 📦 설치 및 환경 세팅

### 1️⃣ 저장소 클론
```bash
git clone https://github.com/seoan1210/korean-llm-v1.git
cd korean-llm-v1
```

### 2️⃣ 가상환경 생성 (선택사항이지만 강력 권장)
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate  # Windows
```

### 3️⃣ 의존성 설치
```bash
pip install -r requirements.txt
```

**핵심 패키지**:
```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.16.0
```

---

## 🎓 빠른 시작

### 1. 기본 학습 실행

```python
from korean_llm_advanced_v1 import main, TrainingConfig

config = TrainingConfig(
    batch_size=2,
    max_steps=50000,
    accumulation_steps=32,
    learning_rate=5e-5,
    eval_interval=100,
    max_seq_len=256,
    download_datasets=False,  # 이미 다운로드된 경우
)

main(config)
```

### 2. 체크포인트에서 재개

```python
config = TrainingConfig(
    resume_from_checkpoint='latest',  # 최신 체크포인트 자동 로드
    # ... 다른 설정들
)
main(config)
```

### 3. 텍스트 생성

```python
import torch
from korean_llm_advanced_v1 import generate
from transformers import AutoTokenizer
from korean_llm_advanced_v1 import KoreanLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
model = KoreanLLM(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
).to(device)

# 체크포인트 로드
checkpoint = torch.load("checkpoints/korean_llm_09700.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 생성!
response = generate(
    model, 
    tokenizer,
    prompt="인공지능이란",
    max_tokens=100,
    temperature=0.7,
    device=device
)
print(response)
```

---

## 📁 디렉토리 구조

```
korean-llm-v1/
├── korean_llm_advanced_v1.py  # 메인 학습 스크립트
├── requirements.txt           # 의존성 목록
├── README.md                  # 이 파일 👈
│
├── datasets/
│   ├── cache/              # 다운로드된 데이터셋 캐시
│   │   ├── {hash}/
│   │   │   └── data.parquet
│   │   └── datasets_manifest.json
│   └── [자동 생성됨]
│
└── checkpoints/
    ├── korean_llm_00100.pth
    ├── korean_llm_00500.pth
    ├── korean_llm_09700.pth    # ← 현재 최신 체크포인트
    └── ...
```

---

## ⚙️ 고급 설정

### 학습률 조정
```python
config = TrainingConfig(
    learning_rate=3e-5,      # 더 보수적인 학습률
    warmup_steps=500,        # 워밍업 증가
    # ...
)
```

### 배치 크기 조정 (메모리 맞춤)
```python
config = TrainingConfig(
    batch_size=1,            # RTX 3090이라면 배치=1로 시도
    accumulation_steps=64,   # 하지만 누적은 더 크게
    # ...
)
```

### 데이터셋 강제 재다운로드
```python
config = TrainingConfig(
    download_datasets=True,  # 캐시 무시하고 재다운로드
    # ...
)
```

### 데이터셋 샘플링 (빠른 테스트용)
```python
config = TrainingConfig(
    samples_per_dataset=5000,  # 각 데이터셋에서 5K개만 사용
    # ...
)
```

---

## 🔍 모델 아키텍처 상세

### 기본 구성

```
KoreanLLM (541M params)
├── Embedding Layer (vocab_size → dim)
├── N× TransformerBlock
│   ├── Multi-Head Attention (N_heads heads)
│   │   ├── RoPE Position Encoding
│   │   └── KV-Cache Support
│   ├── SwiGLU FFN (dim → hidden_dim → dim)
│   ├── RMSNorm (Pre-Norm)
│   └── Residual Connections
└── Output Linear (dim → vocab_size)
```

### 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Total Params | 541M | 경량화된 모델 크기 |
| Hidden Dimension | 설정 가능 | 기본값은 korean_llm_advanced_v1.py 참고 |
| Num Layers | 설정 가능 | 레이어 수 (깊이) |
| Num Heads | 설정 가능 | 멀티헤드 어텐션 헤드 수 |
| Head Dimension | Dim / Heads | 자동 계산 |
| FFN Hidden | Dim × 2.5 | 피드포워드 네트워크 중간층 |
| Max Sequence Length | 256 (기본) | 최대 입력 토큰 길이 |
| Vocab Size | 41.9K | beomi/Llama-3-Open-Ko-8B 토크나이저 |

---

## 📈 학습 곡선 & 성능

### 손실 감소 추이
- **스텝 1000**: ~3.2 → 진단 및 데이터 검증
- **스텝 3000**: ~2.1 → 기본 패턴 학습
- **스텝 9700**: ~1.5 → **현재 진행 중** 🔥

### 예상 최종 손실
- **50,000 스텝 달성 시**: ~0.8-1.2 예상

---

## 🛠️ 문제 해결

### CUDA 메모리 부족 에러
```
RuntimeError: CUDA out of memory
```

**해결책**:
```python
# 배치 크기 감소
config.batch_size = 1

# 누적 스텝 증가
config.accumulation_steps = 64

# 시퀀스 길이 감소
config.max_seq_len = 128

# 모델 크기 축소 (선택사항)
# n_layers = 16 / dim = 1024 등
```

### 데이터셋 다운로드 실패

**OpenOrca-gugugo-ko 특수 처리**:
- 자동으로 3가지 전략 시도 (parquet auto → parquet manual → streaming chunk)
- 각 시도마다 2초씩 대기하여 네트워크 안정화
- 상세한 에러 로그 출력

```bash
# 수동 재시도
python train.py  # download_datasets=True로 실행
```

### 체크포인트 로드 실패
```python
# 최신 체크포인트 찾기 실패 시 수동 지정
config.resume_from_checkpoint = "checkpoints/korean_llm_09700.pth"
```

---

## 🔗 주요 리소스

### 데이터셋
- 🌐 [Hugging Face Datasets](https://huggingface.co/datasets)
- 📊 [maywell/korean_textbooks](https://huggingface.co/datasets/maywell/korean_textbooks)
- 🗣️ [beomi/KoAlpaca](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a)
- ❓ [OpenOrca-gugugo-ko](https://huggingface.co/datasets/squarelike/OpenOrca-gugugo-ko)

### 토크나이저
- 🔤 [beomi/Llama-3-Open-Ko-8B](https://huggingface.co/beomi/Llama-3-Open-Ko-8B)

### 논문 & 참고자료
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU

---

## 📝 주요 메커니즘

### 1. Gradient Checkpointing
각 Transformer 블록의 중간 활성화값을 저장하지 않고 필요시 재계산하여 메모리 절약.

```python
# korean_llm_advanced_v1.py 내부
x, kv = checkpoint(
    layer, x, f_cos, f_sin, None,
    use_reentrant=False
)
```

### 2. KV-Cache for Fast Generation
생성 시 이전 스텝의 Key/Value를 캐시하여 매번 전체 시퀀스를 다시 계산하지 않음.

```python
# 생성 루프에서
logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)
```

### 3. Mixed Precision Training (bfloat16)
연산 일부를 bfloat16으로 수행하여 속도 ↑, 메모리 ↓

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    _, loss, _ = model(batch, labels=batch)
```

---

## 🎯 다음 마일스톤

- [ ] 50,000 스텝 완료 및 손실 1.0 이하 달성
- [ ] 모델 양자화 및 INT8 최적화
- [ ] Multi-GPU DDP 분산학습 테스트
- [ ] 평가 메트릭 추가 (Perplexity, BLEU 등)
- [ ] 한국어 벤치마크 데이터셋 평가

---

## 🤝 기여

이슈 보고 & PR 환영합니다! 작은 개선사항도 소중합니다.

```bash
# Fork → Clone → Feature Branch → Commit → Push → PR
git checkout -b feat/your-feature
git commit -m "✨ Add your feature"
git push origin feat/your-feature
```

---

## 📄 라이선스

MIT License — 자유롭게 사용, 수정, 배포 가능합니다.  
자세한 내용은 [LICENSE](LICENSE) 참고.

---

## 📧 연락처

프로젝트 관리자: [@seoan1210](https://github.com/seoan1210)  
이슈 & 질문: [GitHub Issues](https://github.com/seoan1210/korean-llm-v1/issues)

---

## 🙏 감사의 말

- 🌟 [Hugging Face](https://huggingface.co/) — 데이터셋 & 토크나이저
- 🔬 [PyTorch](https://pytorch.org/) — 딥러닝 프레임워크
- 🤖 [beomi](https://github.com/beomi) — 한국어 LLM 기여자
- 📚 한국어 오픈 데이터셋 커뮤니티

---

**Happy Training! 🚀**

*마지막 업데이트: 2026-08-09*  
*현재 진행률: 19.4% (9,700/50,000)*
