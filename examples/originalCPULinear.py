import torch
import torch.nn as nn
import torch.nn.functional as F

class CPUOffloadLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Store weight on CPU (NOT Parameter, NOT buffer)
        self.weight_cpu = linear.weight.detach().cpu().contiguous()

        # Bias stays as Parameter (so .to(device) works normally)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x):
        # Move weight to GPU on demand
        weight_gpu = self.weight_cpu.to(
            device=x.device,
            dtype=x.dtype,
            non_blocking=True
        )
        return F.linear(x, weight_gpu, self.bias)


def replace_linear_layer_cpu(module: nn.Module):
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(module)}")

    return CPUOffloadLinear(module)