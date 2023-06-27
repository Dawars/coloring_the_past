import os
from pathlib import Path

import numpy as np
import yaml

from nerfstudio.data.utils.colmap_utils import read_points3d_binary

scene_name = "hotel_international"
save_path = Path("/mnt/hdd/3d_recon/neural_recon_w/jena/") / scene_name
colmap_path = save_path / "dense/sparse"

point_path = colmap_path / "points3D.bin"
points_3d = read_points3d_binary(point_path)
points_ori = []
for id, p in points_3d.items():
    if p.point2D_idxs.shape[0] > 2:
        points_ori.append(p.xyz)
points = np.array(points_ori)

sfm_points = points


def bbx_selection(sfm_points):
    """auto scene origin and radius selection

    Args:
        sfm_points (numpy.array): sfm points
    returns:
        bbx (numpy.array): (3 x 2), [min. max]
        origin (numpy.array): (1, 3), [x,y,z]
    """
    bbx = np.concatenate([np.percentile(sfm_points, q=0.00, axis=0).reshape(1, 3),
                          np.percentile(sfm_points, q=100.0, axis=0).reshape(1, 3)], axis=0)
    origin = np.mean(bbx, axis=0)
    return bbx, origin


def generate_config(scene_name: str, save_path: Path, sfm_points):
    # genrate origin and bbx
    bbx, origin = bbx_selection(sfm_points)
    sfm2gt = np.eye(4)
    scale = (np.max(bbx[1] - bbx[0]) / 2).item()
    level = 5

    print(origin.tolist())
    config_dict = {
        'name': scene_name,
        'origin': origin.tolist(),
        'radius': scale * 2,
        'eval_bbx': bbx.tolist(),
        'sfm2gt': sfm2gt.tolist(),
        'min_track_length': 2,
        'eval_bbx_detail': bbx.tolist(),
        'voxel_size': 2 / (2 ** level) * scale - 0.0001
    }

    print(f"config: {config_dict}")

    config_path = save_path / 'config.yaml'
    with open(config_path, 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=False)


generate_config(scene_name, save_path, sfm_points)
