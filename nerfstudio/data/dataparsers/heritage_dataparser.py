# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phototourism dataset parser. Datasets and documentation here: http://phototour.cs.washington.edu/datasets/"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch
import pandas as pd
import yaml
from rich.progress import Console, track
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
    Semantics,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils import colmap_utils
from nerfstudio.data.utils.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from nerfstudio.model_components.ray_samplers import save_points
from nerfstudio.utils.images import BasicImages

CONSOLE = Console(width=120)


def get_masks(image_idx: int, fg_masks, sparse_pts):
    """function to process additional mask information

    Args:
        image_idx: specific image index to work with
        mask: mask data
    """

    # foreground mask
    fg_mask = fg_masks[image_idx]
    fg_mask = BasicImages([fg_mask])

    # sparse_pts
    sparse_points = sparse_pts[image_idx]
    sparse_points = BasicImages([sparse_points])

    return {"fg_mask": fg_mask , "sparse_sfm_points": sparse_points}


@dataclass
class HeritageDataParserConfig(DataParserConfig):
    """Phototourism dataset parser config"""

    _target: Type = field(default_factory=lambda: Heritage)
    """target class to instantiate"""
    data: Path = Path("data/phototourism/trevi-fountain")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to include loading of normal """
    include_sensor_depth: bool = False
    """whether or not to include loading of normal """
    include_sparse_pcd: bool = True
    """whether or not to include sparse pcd from colmap"""
    include_dense_pcd: bool = False
    """whether or not to include dense pcd from colmap"""
    depth_extension: str = '.npy'
    """Extension for depth map including dot (.npy or .png usually)"""
    alpha_color: str = "white"
    """alpha color of background"""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    include_semantics: bool = False
    """whether or not to include loading of semantics data"""
    setting: str = ""
    """Choose tsv file which contains subset of images, e.g: all, clean, facade"""


transient_objects = ['person', 'car', 'bicycle', 'minibike', 'tree', "desk",
                     "blanket", "bed ", "tray", "computer", "swimming pool",
                     "plate", "basket", "glass", "food", "land",
                     ]
label_id_mapping_ade20k = {'airplane': 90,
                           'animal': 126,
                           'apparel': 92,
                           'arcade machine': 78,
                           'armchair': 30,
                           'ashcan': 138,
                           'awning': 86,
                           'bag': 115,
                           'ball': 119,
                           'bannister': 95,
                           'bar': 77,
                           'barrel': 111,
                           'base': 40,
                           'basket': 112,
                           'bathtub': 37,
                           'bed ': 7,
                           'bench': 69,
                           'bicycle': 127,
                           'blanket': 131,
                           'blind': 63,
                           'boat': 76,
                           'book': 67,
                           'bookcase': 62,
                           'booth': 88,
                           'bottle': 98,
                           'box': 41,
                           'bridge': 61,
                           'buffet': 99,
                           'building': 1,
                           'bulletin board': 144,
                           'bus': 80,
                           'cabinet': 10,
                           'canopy': 106,
                           'car': 20,
                           'case': 55,
                           'ceiling': 5,
                           'chair': 19,
                           'chandelier': 85,
                           'chest of drawers': 44,
                           'clock': 148,
                           'coffee table': 64,
                           'column': 42,
                           'computer': 74,
                           'conveyer belt': 105,
                           'counter': 45,
                           'countertop': 70,
                           'cradle': 117,
                           'crt screen': 141,
                           'curtain': 18,
                           'cushion': 39,
                           'desk': 33,
                           'dirt track': 91,
                           'dishwasher': 129,
                           'door': 14,
                           'earth': 13,
                           'escalator': 96,
                           'fan': 139,
                           'fence': 32,
                           'field': 29,
                           'fireplace': 49,
                           'flag': 149,
                           'floor': 3,
                           'flower': 66,
                           'food': 120,
                           'fountain': 104,
                           'glass': 147,
                           'grandstand': 51,
                           'grass': 9,
                           'hill': 68,
                           'hood': 133,
                           'house': 25,
                           'hovel': 79,
                           'kitchen island': 73,
                           'lake': 128,
                           'lamp': 36,
                           'land': 94,
                           'light': 82,
                           'microwave': 124,
                           'minibike': 116,
                           'mirror': 27,
                           'monitor': 143,
                           'mountain': 16,
                           'ottoman': 97,
                           'oven': 118,
                           'painting': 22,
                           'palm': 72,
                           'path': 52,
                           'person': 12,
                           'pier': 140,
                           'pillow': 57,
                           'plant': 17,
                           'plate': 142,
                           'plaything': 108,
                           'pole': 93,
                           'pool table': 56,
                           'poster': 100,
                           'pot': 125,
                           'radiator': 146,
                           'railing': 38,
                           'refrigerator': 50,
                           'river': 60,
                           'road': 6,
                           'rock': 34,
                           'rug': 28,
                           'runway': 54,
                           'sand': 46,
                           'sconce': 134,
                           'screen': 130,
                           'screen door': 58,
                           'sculpture': 132,
                           'sea': 26,
                           'seat': 31,
                           'shelf': 24,
                           'ship': 103,
                           'shower': 145,
                           'sidewalk': 11,
                           'signboard': 43,
                           'sink': 47,
                           'sky': 2,
                           'skyscraper': 48,
                           'sofa': 23,
                           'stage': 101,
                           'stairs': 53,
                           'stairway': 59,
                           'step': 121,
                           'stool': 110,
                           'stove': 71,
                           'streetlight': 87,
                           'swimming pool': 109,
                           'swivel chair': 75,
                           'table': 15,
                           'tank': 122,
                           'television receiver': 89,
                           'tent': 114,
                           'toilet': 65,
                           'towel': 81,
                           'tower': 84,
                           'trade name': 123,
                           'traffic light': 136,
                           'tray': 137,
                           'tree': 4,
                           'truck': 83,
                           'van': 102,
                           'vase': 135,
                           'wall': 0,
                           'wardrobe': 35,
                           'washer': 107,
                           'water': 21,
                           'waterfall': 113,
                           'windowpane': 8}
id_label_mapping_ade20k = {v: k for k, v in label_id_mapping_ade20k.items()}

@dataclass
class Heritage(DataParser):
    """Phototourism dataset. This is based on https://github.com/kwea123/nerf_pl/blob/nerfw/datasets/phototourism.py
    and uses colmap's utils file to read the poses.
    """

    config: HeritageDataParserConfig

    def __init__(self, config: HeritageDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.alpha_color = config.alpha_color

    # pylint: disable=too-many-statements
    def _generate_dataparser_outputs(self, split="train"):

        setting_suffix = '' if self.config.setting == '' else f'_{self.config.setting}'
        config_path = self.data / f"config{setting_suffix}.yaml"
        print(f"Config file: {str(config_path)}")

        with open(config_path, "r") as yamlfile:
            scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

        sfm_to_gt = np.array(scene_config["sfm2gt"])
        gt_to_sfm = np.linalg.inv(sfm_to_gt)
        sfm_vert1 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][0]) + gt_to_sfm[:3, 3]
        sfm_vert2 = gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][1]) + gt_to_sfm[:3, 3]
        bbx_min = np.minimum(sfm_vert1, sfm_vert2)
        bbx_max = np.maximum(sfm_vert1, sfm_vert2)

        with CONSOLE.status(f"[bold green]Reading phototourism images and poses for {split} split...") as _:
            cams = read_cameras_binary(self.data / "dense/sparse/cameras.bin")
            imgs = read_images_binary(self.data / "dense/sparse/images.bin")
            pts3d = read_points3d_binary(self.data / "dense/sparse/points3D.bin")

        img_path_to_id = {}
        file_list = []
        image_list = list(self.data.glob(f"*{setting_suffix}.tsv"))
        if image_list:
            print(f"Found .tsv file for image list {image_list[0]}")
            self.files = pd.read_csv(image_list[0], sep="\t")
            self.files = self.files[~self.files['id'].isnull()]  # remove data without id
            self.files.reset_index(inplace=True, drop=True)
            file_list = list(self.files["filename"])

            for v in imgs.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {}  # {id: filename}
            for filename in list(self.files['filename']):
                if filename not in img_path_to_id:
                    continue
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]
            # for v in imgs.values():
            #     if v.name in file_list:
            #         img_path_to_id[v.name] = v.id
        else:
            raise
            # for _id, cam in cams.items():
            #     img = imgs[_id]
            #     img_path_to_id[img.name] = img.id
            #     file_list.append(img.name)
        # file_list = sorted(file_list)

        # key point depth
        pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4)
        error_array = torch.ones(max(pts3d.keys()) + 1, 1)
        for pts_id, pts in track(pts3d.items(), description="create 3D points", transient=True):
            pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz)
            error_array[pts_id, 0] = torch.from_numpy(pts.error)

        if self.config.include_dense_pcd:
            dense_visibility = defaultdict(list)
            fused_mesh_path = self.data / "dense/fused.ply"
            fused_vis_mesh_path = self.data / "dense/fused.ply.vis"

            dense_pcd = colmap_utils.read_fused(str(fused_mesh_path), str(fused_vis_mesh_path))
            dense_pts3d_array = torch.ones((len(dense_pcd), 4))

            for i, point in enumerate(dense_pcd):
                for img_id in point.visible_image_idxs:
                    dense_visibility[img_id].append(i)
                    dense_pts3d_array[i, :3] = torch.from_numpy(point.position)

        # determine mask extension
        mask_ext = ".npy" if list((self.data / "masks").glob("*.npy")) else ".png"

        poses = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        heights = []
        widths = []
        image_filenames = []
        mask_filenames = []
        semantic_filenames = []
        depth_filenames = []
        sensor_filenames = []
        normal_filenames = []
        sparse_pts = []
        dense_pts = []

        # flip = torch.eye(3)
        # flip[0, 0] = -1.0
        # flip = flip.double()

        for filename in file_list:
            if filename not in img_path_to_id.keys():
                print(f"image {filename} not found in sfm result!!")
                continue
            _id = img_path_to_id[filename]
            img = imgs[_id]
            cam = cams[img.camera_id]

            assert cam.model == "PINHOLE", "Only pinhole (perspective) camera model is supported at the moment"

            pose = torch.cat([torch.tensor(img.qvec2rotmat()), torch.tensor(img.tvec.reshape(3, 1))], dim=1)
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)  # cam pose
            poses.append(torch.linalg.inv(pose))  # cam to world
            fxs.append(torch.tensor(cam.params[0]))
            fys.append(torch.tensor(cam.params[1]))
            cxs.append(torch.tensor(cam.params[2]))
            cys.append(torch.tensor(cam.params[3]))
            heights.append(torch.tensor(cam.height))
            widths.append(torch.tensor(cam.width))

            image_filenames.append(self.data / "dense/images" / img.name)
            mask_filenames.append(self.data / "masks" / img.name.replace(".jpg", mask_ext))
            semantic_filenames.append(self.data / "semantic_maps" / img.name.replace(".jpg", ".npz"))  # todo change to nerfstudio format
            if self.config.include_mono_prior:
                depth_filenames.append(self.data / "depth" / img.name.replace(".jpg", self.config.depth_extension))
                sensor_filenames.append(self.data / "dense" / "stereo" / "depth_maps" / img.name.replace(".jpg", ".jpg.geometric.bin"))
                normal_filenames.append(self.data / "normal" / img.name.replace(".jpg", ".npy"))

            # load sparse 3d points for each view
            # visualize pts3d for each image
            valid_3d_mask = img.point3D_ids != -1
            point3d_ids = img.point3D_ids[valid_3d_mask]
            img_p3d = pts3d_array[point3d_ids]
            img_err = error_array[point3d_ids]
            # img_p3d = img_p3d[img_err[:, 0] < torch.median(img_err)]

            # add dense colmap pcd
            if self.config.include_dense_pcd:
                dense_mask = dense_visibility[img.id]
                dense_img_p3d = dense_pts3d_array[np.array(dense_mask, dtype=int)]
                dense_pts.append(dense_img_p3d)

            if self.config.include_sparse_pcd:
                # weight term as in NeuralRecon-W
                err_mean = img_err.mean()
                weight = 2 * np.exp(-((img_err / err_mean) ** 2))

                img_p3d[:, 3:] = weight

                sparse_pts.append(img_p3d)

        poses = torch.stack(poses).float()
        poses[..., 1:3] *= -1  # flip y,z converting to nerfstudio/opengl camera tr
        fxs = torch.stack(fxs).float()
        fys = torch.stack(fys).float()
        cxs = torch.stack(cxs).float()
        cys = torch.stack(cys).float()
        heights = torch.stack(heights)#.float()
        widths = torch.stack(widths)#.float()

        # filter image_filenames and poses based on train/eval split percentage

        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        i_train = [i for i, filename in enumerate(image_filenames)
                   if self.files.loc[i, 'split'] == 'train']
        i_eval = [i for i, filename in enumerate(image_filenames)
                  if self.files.loc[i, 'split'] == 'test']

        # num_train_images = len(i_train)
        # num_train_images = math.ceil(num_images * self.config.train_split_percentage)
        # num_eval_images = num_images - num_train_images
        # # num_eval_images = 1
        # # num_train_images = num_images - num_eval_images
        # i_all = np.arange(num_images)
        # i_train = np.linspace(
        #     0, num_images - 1, num_train_images, dtype=int
        # )  # equally spaced training images starting and ending at 0 and num_images-1
        # i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        # assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")
        """
        poses = camera_utils.auto_orient_and_center_poses(
            poses, method=self.config.orientation_method, center_poses=self.config.center_poses
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= torch.max(torch.abs(poses[:, :3, 3]))

        poses[:, :3, 3] *= scale_factor * self.config.scale_factor
        # shift back so that the object is aligned?
        poses[:, 1, 3] -= 1
        """

        # normalize with scene radius
        radius = scene_config["radius"]
        scale = 1.0 / (radius * 1.01)  # enlarge the radius a little bit
        origin = np.array(scene_config["origin"]).reshape(1, 3)
        origin = torch.from_numpy(origin)
        poses[:, :3, 3] -= origin
        poses[:, :3, 3] *= scale

        poses, transform = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_poses=False,
        )

        # scale pts accordingly
        for pts in sparse_pts:
            pts[:, :3] -= origin
            pts[:, :3] *= scale  # should be the same as pose preprocessing
            pts[:, :3] = pts[:, :3] @ transform[:3, :3].t() + transform[:3, 3:].t()

        for pts in dense_pts:
            pts[:, :3] -= origin
            pts[:, :3] *= scale  # should be the same as pose preprocessing
            pts[:, :3] = pts[:, :3] @ transform[:3, :3].t() + transform[:3, 3:].t()

        # create occupancy grid from sparse points
        points_ori = []
        min_track_length = scene_config["min_track_length"]
        for _, p in pts3d.items():
            if p.point2D_idxs.shape[0] > min_track_length:
                points_ori.append(p.xyz)
        points_ori = np.array(points_ori)
        save_points("nori_10.ply", points_ori)

        # filter with bbox
        # normalize cropped area to [-1, -1]
        scene_origin = origin.numpy()

        points_normalized = (points_ori - scene_origin) / (bbx_max - bbx_min) * 2
        print(f"{(bbx_max - bbx_min)=}")
        # filter out points out of [-1, 1]
        mask = np.prod((points_normalized > -1), axis=-1, dtype=bool) & np.prod(
            (points_normalized < 1), axis=-1, dtype=bool
        )
        points_ori = points_ori[mask]

        save_points("nori_10_filterbbox.ply", points_ori)

        points_ori = torch.from_numpy(points_ori).float()

        # scale pts accordingly
        points_ori -= origin
        points_ori[:, :3] *= scale  # should be the same as pose preprocessing
        points_ori[:, :3] = points_ori[:, :3] @ transform[:3, :3].t() + transform[:3, 3:].t()

        print(points_ori.shape)

        # expand and quantify

        offset = torch.linspace(-1, 1.0, 3)
        offset_cube = torch.meshgrid(offset, offset, offset)
        offset_cube = torch.stack(offset_cube, dim=-1).reshape(-1, 3)

        voxel_size = scene_config["voxel_size"] / (radius * 1.01)
        offset_cube *= voxel_size  # voxel size
        expand_points = points_ori[:, None, :] + offset_cube[None]
        expand_points = expand_points.reshape(-1, 3)
        save_points("expand_points.ply", expand_points.numpy())

        # filter
        # filter out points out of [-1, 1]
        mask = torch.prod((expand_points > -1.0), axis=-1, dtype=torch.bool) & torch.prod(
            (expand_points < 1.0), axis=-1, dtype=torch.bool
        )
        filtered_points = expand_points[mask]
        save_points("filtered_points.ply", filtered_points.numpy())

        grid_size = 32
        voxel_size = 2.0 / grid_size
        quantified_points = torch.floor((filtered_points + 1.0) * grid_size // 2)

        index = quantified_points[:, 0] * grid_size**2 + quantified_points[:, 1] * grid_size + quantified_points[:, 2]

        offset = torch.linspace(-1.0 + voxel_size / 2.0, 1.0 - voxel_size / 2.0, grid_size)
        x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
        offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        # xyz
        mask = torch.zeros(grid_size**3, dtype=torch.bool)
        mask[index.long()] = True

        points_valid = offset_cube[mask]
        save_points("quantified_points.ply", points_valid.numpy())
        # breakpoint()

        mask = mask.reshape(grid_size, grid_size, grid_size).contiguous()

        # in x,y,z order
        # assumes that the scene is centered at the origin
        scene_box = SceneBox(
            aabb=torch.from_numpy(np.stack([bbx_min, bbx_max]) * scale),
            coarse_binary_gird=mask,
        )

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fxs,
            fy=fys,
            cx=cxs,
            cy=cys,
            width=widths,
            height=heights,
            camera_type=CameraType.PERSPECTIVE,
        )

        # for debug
        # for _ in range(10):
        #    print("==================================================")

        # indices = indices[::20]
        cameras = cameras[torch.tensor(indices)]
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices]
        if sparse_pts:
            sparse_pts = [sparse_pts[i] for i in indices]
        if dense_pts:
            dense_pts = [dense_pts[i] for i in indices]
        if self.config.include_mono_prior:
            depth_filenames = [depth_filenames[i] for i in indices]
            normal_filenames = [normal_filenames[i] for i in indices]
        if self.config.include_sensor_depth:
            sensor_filenames = [sensor_filenames[i] for i in indices]

        assert len(cameras) == len(image_filenames)

        metadata = {
            # "masks": {"func": get_masks, "kwargs": {"fg_masks": fg_masks, "sparse_pts": sfm_points}}
        }
        # --- semantics ---
        if self.config.include_semantics:
            classes = [id_label_mapping_ade20k[i] for i in range(len(id_label_mapping_ade20k))]
            semantics = Semantics(filenames=semantic_filenames, classes=classes, colors=None,
                                  mask_classes=transient_objects)
            metadata["semantics"] = semantics
        # combine points
        sfm_points = []
        for sparse, dense in itertools.zip_longest(sparse_pts, dense_pts, fillvalue=torch.zeros((0, 4), dtype=torch.float32)):
            sfm_points.append(torch.cat((sparse, dense)))


        if self.config.include_mono_prior:
            metadata["depth_filenames"] = depth_filenames
            metadata["depth_unit_scale_factor"] = 1
            metadata["normal_filenames"] = normal_filenames
        if self.config.include_sensor_depth:
            metadata["sensor_filenames"] = sensor_filenames
            metadata["depth_unit_scale_factor"] = 1

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            metadata=metadata,
            dataparser_transform=transform,  # origin substraction not included
            dataparser_scale=scale,
        )

        return dataparser_outputs
