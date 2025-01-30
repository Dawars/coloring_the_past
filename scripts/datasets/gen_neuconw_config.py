import os
from pathlib import Path

import numpy as np
import yaml

def generate_config(scene_root: Path):
    # generate origin and bbx
    # import colmap dense model dir to blender using Photogrammetry Addon
    # Add (axis aligned) box and set position and scale
    # box will cut pcd outside and determin bounding sphere radius
    scene_name = scene_root.name
    origin = np.array([-0.98, -0.44, 0.7])  # position
    bb_scale = np.array([5.08, 1.52, 6.02])  # scale

    sfm2gt = np.eye(4)
    scale = (np.linalg.norm(bb_scale, 2).item())  # this is the radius of the sphere encompassing the bb
    level = 5
    print(scale)

    print(origin.tolist())
    config_dict = {
        'name': scene_name,
        'origin': origin.tolist(),
        'radius': scale,
        'eval_bbx': [(-bb_scale).tolist(), bb_scale.tolist()],  # extents
        'sfm2gt': sfm2gt.tolist(),
        'min_track_length': 2,
        'eval_bbx_detail': [(-bb_scale).tolist(), bb_scale.tolist()],  # extents
        'voxel_size': 2 / (2 ** level) * scale - 0.0001
    }

    print(f"config: {config_dict}")

    config_path = save_path / 'config.yaml'
    with open(config_path, 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=False)

if __name__ == '__main__':
    save_path = Path("/mnt/hdd/nerfstudio/data/theater_mast3r_331")
    generate_config(save_path)
