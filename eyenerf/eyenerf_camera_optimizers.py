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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Type, Union

import torch
import tyro
from torch import Tensor, nn
# from torchtyping import TensorType
from typing_extensions import Literal, assert_never
import numpy as np

# from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
from eyenerf.eyenerf_lie_groups import exp_map_SE3, exp_map_SO3xR3, so3_log_map
from nerfstudio.cameras.camera_optimizers import CameraOptimizer as CameraOptimizer_
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    SchedulerConfig,
)
from nerfstudio.utils import poses as pose_utils
from scipy.spatial.transform import Rotation as R

@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3", "depth"] = "depth"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    position_noise_std: float = 0.0
    """Noise to add to initial positions. Useful for debugging."""

    orientation_noise_std: float = 0.0
    """Noise to add to initial orientations. Useful for debugging."""

    optimizer: AdamOptimizerConfig = AdamOptimizerConfig(lr=6e-4, eps=1e-15)
    """ADAM parameters for camera optimization."""

    scheduler: SchedulerConfig = ExponentialDecaySchedulerConfig(max_steps=10000)
    """Learning rate scheduler for camera optimizer.."""

    param_group: tyro.conf.Suppress[str] = "camera_opt"
    """Name of the parameter group used for pose optimization. Can be any string that doesn't conflict with other
    groups."""


class CameraOptimizer(CameraOptimizer_):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        # we trick the nerfstudio camera optimizer
        actual_mode = config.mode 
        config.mode = "off"
        super().__init__(config, num_cameras, device)
        config.mode = actual_mode
        # print(kwargs)
        self.config = config
        self.num_cameras = num_cameras
        self.device = device

        # Initialize learnable parameters.
        # very hacky below, just for ablation studies
        if self.config.mode == "off":
            pose_adjustment = 0.0 * torch.ones((num_cameras,), device=device) # is zero-init ok? 
            pose_adjustment_x = 0.0 * torch.ones((num_cameras,), device=device)
            pose_adjustment_y = 0.0 * torch.ones((num_cameras,), device=device)

            rotation_component = torch.zeros((num_cameras, 6), device=device)
            rotation_noise_level = kwargs['rotation_noise_level'] 
            if rotation_noise_level> 0.0:
                for i in range(num_cameras):
                    rotation_noise = np.random.uniform(-rotation_noise_level, rotation_noise_level, 3)
                    rot = R.from_euler("XYZ", rotation_noise, degrees=True)
                    rot_mat = torch.from_numpy(rot.as_matrix()).float()
                    log_rot = so3_log_map(rot_mat[None, ...])
                    rotation_component[i, 3:] = log_rot[0]
            rot_init = rotation_component.clone()
            self.register_buffer("pose_adjustment", pose_adjustment)
            self.register_buffer("pose_adjustment_x", pose_adjustment_x)
            self.register_buffer("pose_adjustment_y", pose_adjustment_y)
            self.register_buffer("rotation_component", rotation_component)
            self.register_buffer("rot_init", rot_init)
            
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        elif self.config.mode == "depth":
            # this "parameterization" used below is very silly
            self.pose_adjustment = torch.nn.Parameter(0.0 * torch.ones((num_cameras,), device=device)) # is zero-init ok? 
            self.pose_adjustment_x = torch.nn.Parameter(0.0 * torch.ones((num_cameras,), device=device))
            self.pose_adjustment_y = torch.nn.Parameter(0.0 * torch.ones((num_cameras,), device=device))
            rotation_component = torch.zeros((num_cameras, 6), device=device)
            rotation_noise_level = kwargs['rotation_noise_level'] 
            if rotation_noise_level> 0.0:
                for i in range(num_cameras):
                    rotation_noise = np.random.uniform(-rotation_noise_level, rotation_noise_level, 3)
                    rot = R.from_euler("XYZ", rotation_noise, degrees=True)
                    rot_mat = torch.from_numpy(rot.as_matrix()).float()
                    log_rot = so3_log_map(rot_mat[None, ...])
                    rotation_component[i, 3:] = log_rot[0]
            self.rot_init = rotation_component.clone()
            self.rotation_component = torch.nn.Parameter(rotation_component)

        else:
            assert_never(self.config.mode)
            
        # Initialize pose noise; useful for debugging.
        if config.position_noise_std != 0.0 or config.orientation_noise_std != 0.0:
            assert config.position_noise_std >= 0.0 and config.orientation_noise_std >= 0.0
            std_vector = torch.tensor(
                [config.position_noise_std] * 3 + [config.orientation_noise_std] * 3, device=device
            )
            self.pose_noise = exp_map_SE3(torch.normal(torch.zeros((num_cameras, 6), device=device), std_vector))
        else:
            self.pose_noise = None

    def forward(
        self,
        indices: Tensor["num_cameras"],
    ) -> Tensor["num_cameras", 3, 4]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            outputs.append(exp_map_SO3xR3(self.pose_adjustment[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self.pose_adjustment[indices, :]))
        # else:
        #     assert_never(self.config.mode)

        if self.config.mode in ["SO3xR3", "SE3"]:
            # Apply initial pose noise.
            if self.pose_noise is not None:
                outputs.append(self.pose_noise[indices, :, :])

            # Return: identity if no transforms are needed, otherwise multiply transforms together.
            if len(outputs) == 0:
                # Note that using repeat() instead of tile() here would result in unnecessary copies.
                return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
            return functools.reduce(pose_utils.multiply, outputs)
        elif self.config.mode == "depth":
            # return self.pose_adjustment
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
            
