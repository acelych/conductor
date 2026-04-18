from typing import Tuple
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image

# Use relative path pointing to the data directory
file_path = '../data/cifar100/train.parquet'

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    print("Please modify the script with the correct path to your dataset.")
    exit(1)

data = pd.read_parquet(file_path)
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