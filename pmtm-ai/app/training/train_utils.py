import torch
from transformers import AutoTokenizer, BitsAndBytesConfig


def detect_precision():
    """A100/L4/H100 → bf16; T4/V100 → fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def get_bnb_config(compute_dtype) -> BitsAndBytesConfig:
    """반환할 4비트 양자화 BitsAndBytesConfig 설정을 구성합니다."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def setup_training_env(model_id: str, padding_side: str | None = None):
    """정밀도 감지, 토크나이저 로드, 4비트 양자화 설정을 수행합니다.

    Args:
        model_id (str): pretrained 모델 ID 또는 경로.
        padding_side (str, optional): 토크나이저의 padding_side 설정값.

    Returns:
        tuple: (tokenizer, bnb_config, compute_dtype, use_bf16, use_fp16)
    """
    compute_dtype, use_bf16, use_fp16 = detect_precision()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if padding_side is not None:
        tokenizer.padding_side = padding_side

    bnb_config = get_bnb_config(compute_dtype)

    return tokenizer, bnb_config, compute_dtype, use_bf16, use_fp16
