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
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
import json
import random
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal
import math
import time
import os 

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps

import tinycudann as tcnn
import matplotlib.pyplot as plt
from einops import rearrange


@dataclass
class EyeNeRFModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: EyeNeRFModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    """Whether to randomize the background color."""
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    hidden_dim_transient: int = 64
    """Dimension of hidden layers for transient network"""
    num_levels: int = 16
    """Number of levels of the hashmap for the base mlp."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network. Piecewise is preferred for unbounded scenes."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    orientation_loss_mult: float = 0.0001
    """Orientation loss multiplier on computed normals."""
    pred_normal_loss_mult: float = 0.001
    """Predicted normal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    predict_normals: bool = False
    """Whether to predict normals or not."""
    disable_scene_contraction: bool = True
    """Whether to disable scene contraction or not."""
    use_texture_field: bool = False
    """Whether or not to use a field to represent the iris texture"""
    rot_loss_mult: float = 1.0
    """Rotation loss for iris texture multiplier"""


class EyeNeRFModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: EyeNeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        self.user_gradient_scaling = True

        self.base_dir = "/tmp"

        config_fp = "./config_hash.json"
        with open(config_fp) as config_file:
            config = json.load(config_file)

        self.texture_model_left = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,
            n_output_dims=3,
            encoding_config=config["encoding"],
            network_config=config["network"],
        ).cuda()

        self.texture_model_right = tcnn.NetworkWithInputEncoding(
            n_input_dims=2,
            n_output_dims=3,
            encoding_config=config["encoding"],
            network_config=config["network"],
        ).cuda()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.step_count = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = (
            list(self.field.parameters())
            + list(self.texture_model_left.parameters())
            + list(self.texture_model_right.parameters())
        )
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks



    def get_iris_texture(self, eye_coord_frac, left_eye_mask):
        eye_coord_frac = torch.clip(eye_coord_frac, min=-1.0, max=1.0)
        texture_coord = eye_coord_frac * 0.5 + 0.5
        batch_size = len(texture_coord)
        texture_color = torch.zeros((batch_size, 3)).cuda()
        right_eye_mask = ~left_eye_mask
        texture_color = self.texture_model_left(texture_coord).float()
        # texture_color[left_eye_mask] = self.texture_model_left(texture_coord[left_eye_mask]).float()

        # if torch.sum(right_eye_mask) != 0:
        #     print('running right eye')
        # texture_color[right_eye_mask] = self.texture_model_right(texture_coord[right_eye_mask]).float()
        return texture_color

    def save_full_iris_texture(self):
        # base dir hacked in from 
        save_dir = f"{self.base_dir}/recovered_iris_texture"
        os.makedirs(save_dir, exist_ok=True)
        device = torch.device("cuda:0")
        resolution = [200, 200, 3]

        half_dx = 0.5 / resolution[0]
        half_dy = 0.5 / resolution[1]
        xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=device)
        ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=device)
        xv, yv = torch.meshgrid([xs, ys])

        # save left iris
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        with torch.no_grad():
            texture = self.texture_model_left(xy).reshape(resolution[0], resolution[1], 3)
        minimum = texture.cpu().float().min().item()
        texture_vis = texture.cpu().float().numpy() - minimum
        texture_vis = np.clip(texture_vis, a_min=0.0, a_max=1.0)
        plt.imsave(f"{save_dir}/{time.time()}_left.png", texture_vis)

        # save right iris
        with torch.no_grad():
            texture = self.texture_model_right(xy).reshape(resolution[0], resolution[1], 3)
        minimum = texture.cpu().float().min().item()
        texture_vis = texture.cpu().float().numpy() - minimum
        texture_vis = np.clip(texture_vis, a_min=0.0, a_max=1.0)
        plt.imsave(f"{save_dir}/{time.time()}_right.png", texture_vis)

    def get_outputs(self, ray_bundle: RayBundle, save_state_dict=True):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        if self.user_gradient_scaling:#self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)


        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        if save_state_dict and self.step_count % 10000 == 0:
            torch.save(self.state_dict(), f"{self.base_dir}/{self.step_count}.pt")

        if ray_bundle.metadata and "eye_coord" in ray_bundle.metadata and self.config.use_texture_field:
            if self.step_count % 100 == 0:
                self.save_full_iris_texture()

            eye_coord_frac = (ray_bundle.metadata["eye_coord"].cuda() / math.sqrt(2)).float()  # loss of resolution?
            left_eye_mask = ray_bundle.metadata["left_eye_mask"]

            texture = self.get_iris_texture(eye_coord_frac, left_eye_mask)

            if self.config.rot_loss_mult > 0:
                num_angles = 20
                angles = 2 * np.pi * torch.rand(num_angles, device=self.device)
                cos_t = torch.cos(angles)
                sin_t = torch.sin(angles)
                row1 = torch.stack([cos_t, sin_t], dim=-1)
                row2 = torch.stack([-sin_t, cos_t], dim=-1)
                Rs = torch.stack([row1, row2], dim=-1)

                batch_size = len(eye_coord_frac)
                pts_rotated = torch.matmul(Rs[:,None], eye_coord_frac.unsqueeze(-1)).squeeze(-1) # num_rots x num_pts x 2 x 1 
                pts_rotated = rearrange(pts_rotated, 'r p d -> (r p) d') # num_rots x num_pts x 2  -> num_rots * num_pts x 2
                left_eye_mask_rot = left_eye_mask.unsqueeze(0)
                left_eye_mask_rot = left_eye_mask_rot.repeat(num_angles, 1)
                left_eye_mask_rot = rearrange(left_eye_mask_rot, 'r p -> (r p)')            

                texture_rot = self.get_iris_texture(pts_rotated, left_eye_mask_rot)

                texture_rot = rearrange(texture_rot, '(r p) d -> r p d', p=batch_size, r=num_angles)
            else:
                texture_rot = 0.0
        else:
            texture = 0.0
            texture_rot = 0.0
            

        outputs = {
            "rgb": rgb + texture,
            "texture": texture,
            "texture_rot": texture_rot,
            "accumulation": accumulation,
            "depth": depth,
        }

        if ray_bundle.metadata and "absolute_depth_error" in ray_bundle.metadata: 
            outputs["absolute_depth_error"] = ray_bundle.metadata["absolute_depth_error"]

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        self.step_count += 1
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)

        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"]) 
        if "absolute_depth_error" in outputs: 
            loss_dict["rgb_loss"] += torch.nn.functional.mse_loss(outputs["absolute_depth_error"], torch.zeros_like(outputs["absolute_depth_error"]))
        if self.config.rot_loss_mult > 0.0:
            texture_rot = outputs["texture_rot"]
            # breakpoint()
            p90 = torch.quantile(texture_rot, q=0.9, dim=0).unsqueeze(0)
            p10 = torch.quantile(texture_rot, q=0.1, dim=0).unsqueeze(0)
            pts_std = torch.std(texture_rot, dim=0).unsqueeze(0)
            bad_pts = torch.logical_or(texture_rot > p90 + pts_std, texture_rot < p10 - pts_std)
            bad_pts = torch.any(bad_pts, dim=-1) # num rots x num pts

            pts_mean = torch.mean(texture_rot, dim=0).unsqueeze(0)
            diff = texture_rot - pts_mean
            bad_color = diff[bad_pts]
            rotational_loss = torch.nn.functional.mse_loss(bad_color, torch.zeros_like(bad_color))
            # breakpoint()
            loss_dict["rgb_loss"] += self.config.rot_loss_mult * rotational_loss
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
