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

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchtyping import TensorType


def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    # load mask
    if filepath.suffix == ".npy":
        mask = np.load(filepath)  # (H, W)
        # mask = torch.from_numpy(mask).unsqueeze(-1).bool()  # todo convert to PIL
        pil_mask = Image.fromarray(mask, 'L')
    else:
        pil_mask = Image.open(filepath).convert('L')

    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()

    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype="int64").view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    semantics = torch.from_numpy(np.array(pil_image, dtype="int64"))[..., None]
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :, np.newaxis])


def get_normal_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    camera_to_world: TensorType[3, 4],
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a numpy array `*.npy` of floats in the range of [0., 1.].

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        camera_to_world: Camera to world transformation matrix.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [width, height, 1].
    """

    normal = np.load(filepath)
    c, h, w = normal.shape
    if (h, w) != (height, width):
        normal = cv2.resize(normal, (width, height), interpolation=interpolation)

    # important as the output of omnidata is normalized
    normal = normal * 2.0 - 1.0
    #
    normal = torch.from_numpy(normal).float()

    # transform normal to world coordinate system
    rot = camera_to_world[:3, :3].clone()

    normal_map = normal.reshape(3, -1)
    normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

    normal_map = rot @ normal_map
    normal_map = normal_map.permute(1, 0).reshape(h, w, 3)
    return normal_map
