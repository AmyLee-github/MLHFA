import numpy as np
import math
import torch
from torchvision.transforms import transforms
from PIL import Image


def compute(patch):
    weight, height = patch[0].size
    m = weight
    res = 0
    patch_np = np.array(patch[0]).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch_np[:, :-1, :] - patch_np[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch_np[:-1, :, :] - patch_np[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch_np[:-1, :-1, :] - patch_np[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch_np[1:, :-1, :] - patch_np[:-1, 1:, :]))
    res = diff_horizontal + diff_vertical + diff_diagonal
    return res.sum()


def patch_img(img, img_f, patch_size, height):
    img_width, img_height = img.size
    height = int(height)
    patch_size = int(patch_size)
    num_patch = (height // patch_size) * (height // patch_size)
    patch_list = []
    min_len = min(img_height, img_width)
    rz = transforms.Resize((height, height))
    if min_len < patch_size:
        img = rz(img)
    rp = transforms.RandomCrop(patch_size)
    # 随机生成num_patch个不重复随机种子
    seeds = np.random.choice(100000, num_patch, replace=False)
    for i in range(num_patch):
        seed = seeds[i]
        torch.random.manual_seed(seed)
        rp_img = rp(img)
        torch.random.manual_seed(seed)
        rp_img_f = rp(img_f)
        patch_list.append([rp_img, rp_img_f])
    patch_list.sort(key=lambda x: compute(x), reverse=False)
    new_img, new_img_f = patch_list[0][0], patch_list[0][1]
    return new_img, new_img_f
