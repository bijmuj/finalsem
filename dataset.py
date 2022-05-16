from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torch.utils.data import Dataset


class Vimeo90KDataset(Dataset):
    def __init__(
        self,
        base_path: str = "dataset/vimeo_septuplet/sequences/",
        skip_frames: int = 0,
        scale_factor: int = 1,
        transforms=None,
    ):
        """Dataset class for Vimeo90K septuplet sequences.

        Args:
            base_path (str): Path to Vimeo90K dataset
            skip_frames (int, optional): Number of frames to skip. Defaults to 0.
            scale_factor (int, optional): Factor to downscale by. Defaults to 1.
        """
        self.skip_frames = skip_frames
        self.scale_factor = scale_factor
        self.transforms = transforms
        self.sequence_paths = []
        folders = glob(f"{base_path}/*/*")
        for folder in folders:
            contents = sorted(list(glob(f"{folder}/*.png")))
            self.sequence_paths.append(contents)

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, index: int):
        """Returns ground truth and partially transformed sequences of images

        Args:
            index (int): Index for DataLoader.

        Returns:
            tuple[torch.tensor, torch.tensor]: Index 0 are the downscaled and frame skipped images.
                    Index 1 are the ground truth images.
        """
        paths = self.sequence_paths[index]

        imgs = [np.array(Image.open(path)) for path in paths]
        imgs = np.array(imgs).astype(np.float32)
        imgs /= 255.0
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        if self.transforms:
            imgs = self.transforms(imgs)

        inputs = torch.tensor(imgs.copy())
        if self.skip_frames:
            inputs = inputs[:: self.skip_frames + 1]

        if self.scale_factor != 1:
            inputs = interpolate(inputs, scale_factor=self.scale_factor, mode="bicubic")

        return inputs, torch.tensor(imgs)
