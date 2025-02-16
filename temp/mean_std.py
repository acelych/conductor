from typing import Tuple
from tqdm import tqdm

import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image

data = pd.read_parquet(r'C:\Workspace\DataScience\ComputerVision\conductor\temp\cifar100\train-00000-of-00001.parquet')
mean = [0, 0, 0]
ssd = [0, 0, 0]
size = [0, 0, 0]

def welford_mean_ssd(mean: float, ssd: float, size: int, data: np.ndarray) -> Tuple[float, float, int]:
    # calculate mean & sum of squared diff in one-pass
    mean_ = data.mean()
    mean_new = (size * mean + data.size * mean_) / (size + data.size)
    ssd_new = ssd + np.sum((data - mean_) ** 2)
    size_new = size + data.size
    return mean_new, ssd_new, size_new

for row in tqdm(data.iterrows(), total=len(data)):
    img = row[1]['img']['bytes']
    img = Image.open(BytesIO(img))
    img = (np.array(img) / 255.0).transpose((2, 0, 1))
    
    for i, channel in enumerate([img[0, :, :], img[1, :, :], img[2, :, :]]):
        mean[i], ssd[i], size[i] = welford_mean_ssd(mean[i], ssd[i], size[i], channel)

mean = np.asarray(mean)
std = (np.asarray(ssd) / (np.asarray(size) - 1)) ** 0.5
print(mean, std)