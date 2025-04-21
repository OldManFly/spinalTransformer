from typing import Optional

import torch.nn as nn



def get_norm_layer(name:tuple | str, dim: int | None=1, channels: int | None=1) :
    """
    Returns a normalization layer based on the specified type.
    """    

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    elif norm_type == "batchnorm":
        return nn.BatchNorm2d(dim)
    elif norm_type == "instancenorm":
        return nn.InstanceNorm2d(dim)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")