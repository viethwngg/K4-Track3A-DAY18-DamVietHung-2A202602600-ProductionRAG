"""Optional CPU optimization for large BGE models on the lab's 8 GB machine."""

import os
import warnings


def optimize_cpu_model(model) -> bool:
    import torch

    if (
        os.getenv("RAG_CPU_INT8", "1") != "1"
        or next(model.parameters()).device.type != "cpu"
    ):
        return False
    # Quantize only supported Linear layers, in place to avoid a full model copy.
    # Embeddings, normalization layers and activations remain floating point.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="torch.quantize_per_tensor.*")
        torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        )
    model._lab_cpu_int8 = True
    return True
