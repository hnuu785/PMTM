# PMTM SVS runtime

PMTM의 8마디 가이드 랩을 OpenUtau DiffSinger 보이스뱅크로 렌더링하는 별도 Python 3.8 런타임입니다.
백엔드 환경과 `diffsinger-utau`의 오래된 PyTorch/ONNX 의존성을 분리하기 위해 독립 가상환경을 사용합니다.

## 설치

```bash
cd pmtm-svs
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

기본 requirements는 macOS/CPU 개발 환경용입니다. NVIDIA 서버에서는 설치된 CUDA·cuDNN과 호환되는 ONNX Runtime GPU 패키지로 교체합니다.

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.16.3
```

`diffsinger-utau 0.2.1`이 ONNX Runtime 1.16 계열을 요구하므로 CUDA 서버도 먼저 1.16.3으로 검증합니다. 서버 CUDA 버전과 맞지 않으면 보이스뱅크 검증과 별도로 런타임 업그레이드 호환성을 확인해야 합니다.

macOS에서는 사용 가능한 경우 CoreML, NVIDIA 환경에서는 CUDA, 그 외에는 CPU로 자동 폴백합니다.
실제 실행 프로바이더는 설치된 ONNX Runtime 빌드에 따라 달라집니다.

## 보이스뱅크

보이스뱅크 파일은 저장소에 포함하지 않습니다. 각 배포자의 약관에 동의하고 직접 내려받은 뒤 다음처럼 배치합니다.

```text
pmtm-svs/voicebanks/
  potg/
  kitane/
  rang/
  lunar/
```

각 디렉터리는 `diffsinger-utau`가 읽을 수 있는 OpenUtau DiffSinger 구조여야 하며, 호환 vocoder 설정도 포함해야 합니다.
서비스 또는 상업적 이용 전에는 반드시 각 보이스뱅크 제작자의 허가를 확인해야 합니다.

환경변수로 위치와 장치를 바꿀 수 있습니다.

```bash
DIFFSINGER_PYTHON_PATH=../pmtm-svs/.venv/bin/python
DIFFSINGER_VOICEBANK_ROOT=../pmtm-svs/voicebanks
DIFFSINGER_DEVICE=auto
```

`DIFFSINGER_DEVICE`는 `auto`, `cuda`, `mps`, `cpu` 중 하나입니다. 여기서 `mps`는 ONNX Runtime의 CoreML 실행 프로바이더를 선택하는 PMTM 내부 명칭입니다.
