import torchio as tio
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

def pad_to_target_size(image, target_size=(880, 880, 15), pad_value=0):
    """
    Pads a 3D image tensor to a target size.

    Args:
        image (torch.Tensor): Input image tensor (H, W, D) or (C, H, W, D).
        target_size (tuple): Target size (target_H, target_W, target_D).

    Returns:
        torch.Tensor: Padded image tensor.
    """

    # image shape: (H, W, D) or (C, H, W, D)
    # target_size: (target_H, target_W, target_D)

    if len(image.shape) == 3:
        image = image.unsqueeze(0)  # Add channel dimension

    padding = []
    for i in range(len(target_size)):
        diff = target_size[i] - image.shape[i + 1]
        pad_before = diff // 2
        pad_after = diff - pad_before
        padding.extend((pad_before, pad_after))

    padding = tuple(reversed(padding))

    padded_image = F.pad(image, padding, mode='constant', value=pad_value)
    padded_image = padded_image.squeeze(0)

    assert padded_image.shape == target_size, f"Padding failed: {image.shape} -> {padded_image.shape}, target_size={target_size}, padding={padding}"
    return padded_image
class spinalLoader(Dataset):
    def __init__(self, data_dir, target_size=None, transform=None, num_classes=None):  # Add num_classes
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes  # Store num_classes
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = nib.load(image_path).get_fdata()
        image = torch.from_numpy(image).float()
        label_path = image_path.replace('MR', 'Mask')
        try:
            label = nib.load(label_path).get_fdata()
            label = torch.from_numpy(label).long()  # 重要：確保 label 是 long 類型
        except FileNotFoundError:
            label = torch.zeros(image.shape, dtype=torch.long)

        if self.target_size:
            image = pad_to_target_size(image, self.target_size)
            label = pad_to_target_size(label, self.target_size, pad_value=0)

        # 如果使用 Dice Loss，需要將 label 轉換為 one-hot 編碼
        if self.num_classes is not None and self.num_classes > 1:  # 只有當 num_classes > 1 才轉換
            if len(label.shape) == 3:  # If label is [H, W, D]
                label = label.unsqueeze(0)  # Add batch dimension: [1, H, W, D]
            label = label.permute(0, 4, 1, 2, 3)  # Permute BEFORE one_hot
            label = F.one_hot(label, num_classes=self.num_classes).float()

        return image, label


#### Check image dimentaion ####

if __name__ == '__main__':
    # ... (Your testing code - you can adapt it to test the padding)
    pass