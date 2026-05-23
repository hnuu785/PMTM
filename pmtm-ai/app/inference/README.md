# Inference

이 디렉터리에는 서비스에서 호출할 추론 전용 코드를 둡니다.

예시:
- `generate.py`
- `load_model.py`
- `prompt_builder.py`

## CLI

로컬 어댑터로 가사를 생성하려면:

```bash
python -m app.inference.generate \
  --adapter /Users/cho/Downloads/models/grpo_rap_qwen \
  --artist "Tablo" \
  --bpm 90 \
  --energy 0.65 \
  --danceability 0.70 \
  --loudness -6.0 \
  --valence 0.50 \
  --bars 8
```

`--adapter`를 생략하면 기본값으로 `models/grpo_rap_qwen`을 사용합니다.
