"""
Generate masks from voxel and segmentation to skip rays
Add the NeuranRecon-W codebase to the PYTHONPATH
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from datasets import PhototourismDataset
from datasets.mask_utils import label_id_mapping_ade20k

DEBUG = True

root_dir = Path("/mnt/hdd/3d_recon/neural_recon_w/nepszinhaz_all_disk/split_0")
semantic_map_path = "semantic_maps"

kwargs = {
    "root_dir": str(root_dir),
    "img_downscale": 1,
    "val_num": 1,
    "semantic_map_path": semantic_map_path,
    "with_semantics": True,

}

val_dataset = PhototourismDataset(split="test_train", use_cache=False, **kwargs)

excluded_labels = ["desk",
                   "blanket",
                   "bed ",
                   "tray",
                   "computer",
                   "person",
                   "swimming pool",
                   "plate",
                   "basket",
                   "glass",
                   "car",
                   "minibike",
                   "food",
                   "land",
                   "bicycle",
                   ]

(root_dir / "masks").mkdir(exist_ok=True)
sfm_octree = val_dataset.get_octree(device=0, expand=1, radius=1)
for data in tqdm(val_dataset):
    # print(data)

    h, w = data['img_wh']
    shape = (w, h)

    id_ = int(data["ts"][0])

    rays_o, rays_d = data["rays"][..., :3], data["rays"][..., 3:6]
    image_name = val_dataset.image_paths[id_].split(".")[0]
    _, _, valid_mask = val_dataset.near_far_voxel(
        sfm_octree, rays_o, rays_d, image_name
    )

    voxel_mask = valid_mask.detach().cpu().numpy().reshape(shape)
    semantic_mask = data['semantics'].detach().cpu().numpy().reshape(shape)

    mask = voxel_mask
    for label_name in excluded_labels:
        mask[semantic_mask == label_id_mapping_ade20k[label_name]] = False

    # plt.imshow(mask, cmap='gray')
    # plt.show()

    # image_name = data["image_name"]
    image_name = val_dataset.image_paths[id_]
    save_path = (root_dir / "masks" / image_name)
    np.save(save_path.with_suffix('.npy'), mask)
    if DEBUG:
        im_mask = Image.fromarray(mask, 'L')
        im_mask.save(save_path.with_suffix('.png'))
        # plt.imshow(im_mask, cmap='gray')
        # plt.show()
