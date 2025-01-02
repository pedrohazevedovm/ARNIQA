import pandas as pd
import numpy as np


from data.dataset_base_iqa import IQADataset

class Koniq10kDataset(IQADataset):
    """
    KonIQ-10k dataset
    """
    def __init__(self,
                 root: str,
                 phase: str = 'train',
                 split_idx: int = 0,
                 crop_size: int = 224):
        mos_type = "mos"
        mos_range = (1, 100)
        is_synthetic = False
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase,
                         split_idx=split_idx, crop_size=crop_size)
        scores_csv = pd.read_csv(self.root / "koniq10k_scores_and_distributions.csv")
        scores_csv = scores_csv[["image_name", "MOS_zscore"]]

        self.images = scores_csv["image_name"].values.tolist()
        self.images = np.array([self.root / "512x384" / el for el in self.images])

        self.mos = np.array(scores_csv["MOS_zscore"].values.tolist())

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[self.split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # Remove the padding (i.e. -1 indexes)
            self.images = self.images[split_idxs]
            self.mos = self.mos[split_idxs]
