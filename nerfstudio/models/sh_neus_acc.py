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
Implementation of NeuS with empty space skipping and Sh encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.models.neus import NeuSModel
from nerfstudio.models.neus_acc import NeuSAccModelConfig


@dataclass
class SHNeuSAccModelConfig(NeuSAccModelConfig):
    """UniSurf Model Config"""

    _target: Type = field(default_factory=lambda: SHNeuSAccModel)
    """Sky segmentation normal consistency loss multiplier."""


class SHNeuSAccModel(NeuSModel):
    """SHNeuSAcc model

    Args:
        config: SHNeuSAcc configuration to instantiate model
    """

    config: SHNeuSAccModelConfig

    def __init__(self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, **kwargs):
        super().__init__(config, scene_box, num_train_data, **kwargs)

        self.position_encoding = HashEncoding()
        self.direction_encoding = SHEncoding()

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        return callbacks
