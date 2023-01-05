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
Implementation of osr nerf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.field_heads import *
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.fields.osr_nerf_field import OSRNeRFField
from nerfstudio.fields.sh_field import SHField
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler, LinearDisparitySampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
    NormalsRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc
from nerfstudio.utils.math import components_from_spherical_harmonics


@dataclass
class OSRNerfModelConfig(ModelConfig):
    """Outdoor Scene Relighting Model Config"""

    _target: Type = field(default_factory=lambda: OSRNeRFModel)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""

    base_mlp_num_layers = 8
    base_mlp_layer_width = 256
    head_mlp_num_layers = 1
    head_mlp_layer_width = 128

    # use_viewdirs = False
    # activation = relu

    # N_anneal = 30000
    # N_anneal_min_freq = 8
    # N_anneal_min_freq_viewdirs = 4


class OSRNeRFModel(Model):
    """Outdoor Scene Relighting NeRF model

    Args:
        config: OSR NeRF configuration to instantiate model
    """

    def __init__(
        self,
        config: OSRNerfModelConfig,
        # metadata: Dict,
        **kwargs
    ) -> None:
        # assert "num_cameras" in metadata.keys() and isinstance(metadata["semantics"], Semantics)

        self.field_coarse: OSRNeRFField = None
        self.field_fine: OSRNeRFField = None
        self.field_coarse_bg: NeRFField = None
        self.field_fine_bg: NeRFField = None
        self.sh_field: SHField = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=12, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field_coarse = OSRNeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.base_mlp_num_layers,  # 8
            base_mlp_layer_width=self.config.base_mlp_layer_width,  # 256
            head_mlp_num_layers=self.config.head_mlp_num_layers,  # 1,
            head_mlp_layer_width=self.config.head_mlp_layer_width,  # 128,
            skip_connections=(4,),
            field_heads=(RGBFieldHead(),),
            extra_head_in_dims={FieldHeadNames.RGB: direction_encoding.get_out_dim(),},
        )
        self.field_coarse_bg = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.base_mlp_num_layers,  # 8
            base_mlp_layer_width=self.config.base_mlp_layer_width,  # 256
            head_mlp_num_layers=self.config.head_mlp_num_layers,  # 1,
            head_mlp_layer_width=self.config.head_mlp_layer_width,  # 128,
            skip_connections=(4,),
        )
        self.field_fine = OSRNeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.base_mlp_num_layers,  # 8
            base_mlp_layer_width=self.config.base_mlp_layer_width,  # 256
            head_mlp_num_layers=self.config.head_mlp_num_layers,  # 1,
            head_mlp_layer_width=self.config.head_mlp_layer_width,  # 128,
            skip_connections=(4,),
            field_heads=(RGBFieldHead(), ShadowFieldHead()),  # albedo, shadow
            extra_head_in_dims={FieldHeadNames.RGB: direction_encoding.get_out_dim(),
                                FieldHeadNames.SHADOW: 9},
        )
        self.field_fine_bg = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=self.config.base_mlp_num_layers,  # 8
            base_mlp_layer_width=self.config.base_mlp_layer_width,  # 256
            head_mlp_num_layers=self.config.head_mlp_num_layers,  # 1,
            head_mlp_layer_width=self.config.head_mlp_layer_width,  # 128,
            skip_connections=(4,),
        )

        self.sh_field = SHField(1227)  # todo

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        # self.sampler_disparity = LinearDisparitySampler(num_samples=self.config.num_bg_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_normals = NormalsRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")

        param_groups["fields"] = list(self.field_fine.parameters()) + list(self.sh_field.parameters())
        param_groups["proposal_networks"] = list(self.field_coarse.parameters())
        param_groups["field_background"] = list(self.field_coarse_bg.parameters()) + list(self.field_fine_bg.parameters())

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):

        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_coarse = self.sampler_uniform(ray_bundle)
        # ray_samples_bg = self.sampler_disparity(ray_bundle)

        view_dir_encoded_coarse = self.field_coarse.direction_encoding(ray_samples_coarse.frustums.directions)
        # coarse field:
        field_outputs_coarse = self.field_coarse.forward(ray_samples_coarse,
                                                         extra_head_inputs={FieldHeadNames.RGB: view_dir_encoded_coarse,})
        weights_coarse = ray_samples_coarse.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_coarse)
        # todo add bg

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_coarse, weights_coarse)

        sh = self.sh_field.forward(ray_samples_pdf)[FieldHeadNames.SH]
        num_components = sh.shape[-1] // 3
        sh = sh.view(*sh.shape[:-1], 3, num_components)  # 3*9
        sh_coeffs_gray = sh[..., 0, :] * 0.2126 + \
                         sh[..., 1, :] * 0.7152 + \
                         sh[..., 2, :] * 0.0722
        sh_coeffs_gray = torch.unsqueeze(sh_coeffs_gray, 1).tile(1, ray_samples_pdf.shape[1], 1)
        # fine field:
        view_dir_encoded_fine = self.field_fine.direction_encoding(ray_samples_pdf.frustums.directions)
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf,
                                                     compute_normals=True,
                                                     extra_head_inputs={FieldHeadNames.RGB: view_dir_encoded_fine,
                                                                        FieldHeadNames.SHADOW: sh_coeffs_gray})
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        albedo_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )

        shadow_ = field_outputs_fine[FieldHeadNames.SHADOW]
        shadow_ = shadow_.tile([1, 1, 3])
        shadow = self.renderer_rgb(
            rgb=shadow_,
            weights=weights_fine,
        )

        normals = self.renderer_normals(
            normals=field_outputs_fine[FieldHeadNames.NORMALS],
            weights=weights_fine,
        )

        components = components_from_spherical_harmonics(levels=self.sh_field.levels+1, directions=normals)

        env_lighting = sh * components[..., None, :]  # [..., num_samples, 3, sh_components]
        env_lighting = torch.sum(env_lighting, dim=-1) + 0.5  # [..., num_samples, 3]  # todo +1/2?

        rgb_fine = albedo_fine * shadow * env_lighting


        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "albedo_fine": albedo_fine,
            "env_lighting": env_lighting,
            "shadow": shadow,
            "normal_map": normals,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb_coarse"].device
        image = batch["image"].to(device)

        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])
        shadow_reg = torch.mean((1 - outputs['shadow']) ** 2)

        loss_dict = {"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine,
                     "shadow_reg": shadow_reg}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]

        shadow = outputs["shadow"]
        env_lighting = outputs["env_lighting"] / outputs["env_lighting"].max()
        normal_map = outputs["normals"]

        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_osr = torch.cat([normal_map, env_lighting, shadow], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "osr": combined_osr, "depth": combined_depth}
        return metrics_dict, images_dict
