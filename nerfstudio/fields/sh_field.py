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

"""
Spherical Harmonics field.
"""


from typing import Optional

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field


class SHField(Field):
    """A Spherical Harmonics module.

    Args:
        num_images: number of images to creates SH params for
        levels: levels of spherical harmonics
    """

    def __init__(
        self,
        num_images: int,
        levels: int=2,
    ) -> None:
        super().__init__()
        self.num_images = num_images
        self.levels = levels

        self.params = nn.ParameterList([nn.Parameter(torch.tensor([
            [2.9861e+00, 3.4646e+00, 3.9559e+00],
            [1.0013e-01, -6.7589e-02, -3.1161e-01],
            [-8.2520e-01, -5.2738e-01, -9.7385e-02],
            [2.2311e-03, 4.3553e-03, 4.9501e-03],
            [-6.4355e-03, 9.7476e-03, -2.3863e-02],
            [1.1078e-01, -6.0607e-02, -1.9541e-01],
            [7.9123e-01, 7.6916e-01, 5.6288e-01],
            [6.5793e-02, 4.3270e-02, -1.7002e-01],
            [-7.2674e-02, 4.5177e-02, 2.2858e-01]
        ],
            # names=('DIM', 'CH'),
            dtype=torch.float32)) for _ in range(num_images)])



    def get_density(self, ray_samples: RaySamples):
        pass

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}


    def forward(self, ray_samples: RaySamples):
        """Evaluates the spherical harmonics for points along the ray based on camera id.

        Args:
            ray_samples: Samples to evaluate field on.
        """

        field_outputs = self.get_outputs(ray_samples)
        camera_id_shape = ray_samples.camera_indices.shape
        # one sample per ray
        field_outputs[FieldHeadNames.SH] = torch.stack([self.params[i] for i in ray_samples.camera_indices[..., 0, 0]], dim=0)
        field_outputs[FieldHeadNames.SH] = field_outputs[FieldHeadNames.SH].reshape(camera_id_shape[0], 3*9)
        return field_outputs

