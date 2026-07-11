"""Torch device selection helpers for local inference."""


def select_inference_device(torch_module):
    if torch_module.cuda.is_available():
        return "cuda", torch_module.float16
    if (
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    ):
        return "mps", torch_module.float16
    return "cpu", torch_module.float32


def move_model_to_device(model, device: str):
    if device == "mps":
        return model.to("mps")
    return model


def model_device(model):
    return next(model.parameters()).device
