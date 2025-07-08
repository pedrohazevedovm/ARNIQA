import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


from data.dataset_base_iqa import IQADataset
from utils.utils_data import resize_crop, center_corners_crop


class UHDIQADataset(IQADataset):
    """
    UHD-IQA dataset
    """
    def __init__(self,
                 root: str,
                 phase: str = 'train',
                 split_idx: int = 0,
                 crop_size: int = 224):
        mos_type = 'mos'
        mos_range = (0, 1)
        is_synthetic = False
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase,
                         split_idx=split_idx, crop_size=crop_size)
        scores_csv = pd.read_csv(self.root / "uhd-iqa-metadata.csv")
        scores_csv = scores_csv[["image_name", "quality_mos"]]

        self.images = scores_csv["image_name"].values.tolist()
        self.images = np.array([self.root / "database" / el for el in self.images])

        self.mos = np.array(scores_csv["quality_mos"].values.tolist())
        self.split_idx = 0

        if self.phase != 'all':
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[self.split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))
            self.images = self.images[split_idxs]
            self.mos = self.mos[split_idxs]

        self.target_size = 512

    def __getitem__(self, index: int) -> dict:
        img = Image.open(self.images[index]).convert('RGB')

        width, height = img.size
        aspect_ratio = width / height
        if width < height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        img = img.resize((new_width, new_height), Image.BICUBIC)
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        crops = center_corners_crop(img, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]
        img = torch.stack(crops, dim=0)
        crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        img_ds = torch.stack(crops_ds, dim=0)

        img = self.normalize(img)
        img_ds = self.normalize(img_ds)
        mos = self.mos[index]
        return {"img": img, "img_ds": img_ds, "mos": mos}

    def get_split_indices(self, split: int, phase: str):
        return self.splits[phase][0]
