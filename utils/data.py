import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image
import io
import av
import torchvision
import decord as de
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import lru_cache
from glob import glob


def generate_seq(count, k):
    ratio = count / k
    return list(map(lambda i: max(0, int(i * ratio)), range(0, k)))


def getK(arr, k=16):
    out = []
    ratio = len(arr) / k

    for i in range(k):
        out.append(arr[int(i * ratio)])
    return out


def read_video(root, frames):
    try:
        vr = de.VideoReader(root, ctx=de.cpu())
        total = len(vr)
        if total > 0:
            idx = np.linspace(0, total - 1, num=frames, dtype=int)
            return [Image.fromarray(vr[i].asnumpy()) for i in idx]
        else:
            print(f"Video is empty: {root}")
            return None
    except Exception as e:
        print(f"Read error of video {root}, {e}")
        return None


# Kinetics-400
class Kinetics400(Dataset):

    def __init__(self, labels, root_dir, preprocess=None, frames=16, per_sample=1):

        assert per_sample > 0
        with open(labels, "r") as f:
            lines = f.readlines()

        self.src = []
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])
                    self.src.append((filename, label))
                else:
                    print(f"Warning: Invalid line format at line {line_num}: {line}")
            except ValueError as e:
                print(f"Warning: Cannot parse label at line {line_num}: {line}, error: {e}")

        print(f"Successfully loaded {len(self.src)} samples from {labels}")
        self.root_dir = root_dir
        self.frames = frames
        self.preprocess = preprocess
        self.per_sample = per_sample

    def __len__(self):
        return len(self.src) * self.per_sample

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        original_idx = idx
        max_attempts = len(self.src)
        attempts = 0

        while attempts < max_attempts:
            try:
                idx = idx // self.per_sample
                id, label = self.src[idx]

                full_path = f'{self.root_dir}/{id}'
                imgs = read_video(full_path, self.frames)

                if imgs is None or len(imgs) == 0:
                    print(f"Warning: Video read failed or empty: {full_path}, trying next sample")

                    original_idx = (original_idx + 1) % (len(self.src) * self.per_sample)
                    idx = original_idx
                    attempts += 1
                    continue


                if self.preprocess is not None:
                    imgs = [self.preprocess(img).unsqueeze(0) for img in imgs]
                    imgs = torch.cat(imgs)

                patient_prefix = id

                return imgs, int(label), patient_prefix

            except Exception as e:
                print(f"Error processing video {full_path}: {e}, trying next sample")
                original_idx = (original_idx + 1) % (len(self.src) * self.per_sample)
                idx = original_idx
                attempts += 1

        print(f"Warning: All samples failed, returning dummy data for idx {original_idx}")
        dummy_imgs = torch.zeros(self.frames, 3, 224, 224)
        return dummy_imgs, 0, "unknown"
