from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, base_path, skip_frames=0, scale_factor=1):
        self.skip_frames = skip_frames
        self.scale_factor = scale_factor

        self.sequence_paths = []
        folders = glob(f"{base_path}/*/*")
        for folder in folders:
            contents = list(glob(f"{folder}/*.png"))
            self.sequence_paths.append(contents)

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, index):
        paths = self.sequence_paths[index]

        imgs = [np.array(Image.open(path)) for path in paths]
        imgs = np.array(imgs).astype(np.float32)
        imgs /= 255.0
        imgs = np.transpose(imgs, (0, 3, 1, 2))

        if self.skip_frames:
            inputs = torch.tensor(imgs[:: self.skip_frames + 1].copy())
        else:
            inputs = torch.tensor(imgs.copy())

        if self.scale_factor != 1:
            inputs = interpolate(inputs, scale_factor=self.scale_factor, mode="bicubic")

        return inputs, torch.tensor(imgs)
