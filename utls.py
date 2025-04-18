import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2).sum(dim=2)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()