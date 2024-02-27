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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
import os

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class EyenerfDataParserConfig(DataParserConfig):
    """Eyenerf dataset parser config"""

    _target: Type = field(default_factory=lambda: Eyenerf)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    scene_bound: float = 1.0
    """aabb/scenebox size"""
    used_eye: str = 'both'
    """which eyes to use in training ['left', 'right', 'both']"""


@dataclass
class Eyenerf(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: EyenerfDataParserConfig

    def __init__(self, config: EyenerfDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.sb = config.scene_bound
        self.used_order = config.used_eye

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        mask_filenames = []
        poses = []
        ellipse_params = []
        focal_lengths = []
        order = []
        for frame in meta["frames"]:

            if self.used_order == 'both':
                if "order" in frame:
                    order.append(frame["order"] == "left_eye")
                else:
                    order.append(True) # left_eye default
            else:
                if "order" not in frame:
                    raise ValueError("""The json does not include eye order information.
                                     Re-run the data processing script or use the option 
                                     'both' for used_eye""")

                if frame["order"] != self.used_order:
                    continue
                order.append(frame["order"] == "left_eye")


            fp = frame["file_path"]

            if os.path.isabs(fp) or os.path.exists(fp):
                fname = fp
            else:
                fname = self.data / Path(fp.replace("./", ""))

            image_filenames.append(fname)
            if "mask_path" in frame:
                mp = frame["mask_path"]
                if os.path.isabs(mp) or os.path.exists(mp):
                    mask_fname = mp
                else:
                    mask_fname = self.data / Path(mp.replace("./", ""))

                mask_filenames.append(mask_fname)
            if "ellipse_params" in frame:
                ellipse_params.append(frame["ellipse_params"])
            if "fl_x" in frame:
                focal_lengths.append(frame["fl_x"])
            poses.append(np.array(frame["transform_matrix"]))


        poses = np.array(poses).astype(np.float32)


        if "camera_angle_x" in meta:
            if "png" not in image_filenames[0]:
                temp_fn = image_filenames[0] + ".png"
            else:
                temp_fn = image_filenames[0]
            img_0 = imageio.imread(temp_fn)
            image_height, image_width = img_0.shape[:2]
            camera_angle_x = float(meta["camera_angle_x"])
            focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        elif "fl_x" in meta:  # real data
            focal_length = meta["fl_x"]
            image_height = meta["image_height"]
            image_width = meta["image_width"]
        else:
            image_height = meta["image_height"]
            image_width = meta["image_width"]

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        sb = self.sb

        

        if len(order) > 0:
            order = torch.tensor(order, dtype=torch.bool)
        else:
            order = None


        # TODO: be careful with scenebox
        scene_box = SceneBox(aabb=torch.tensor([[-sb, -sb, -sb], [sb, sb, sb]], dtype=torch.float32))
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=torch.tensor(focal_lengths) if len(focal_lengths) > 0 else focal_length,
            fy=torch.tensor(focal_lengths) if len(focal_lengths) > 0 else focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )
        # hacky monkeypatching
        cameras.order = order

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames if split == "train" else None,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata={"ellipse_params": ellipse_params},
        )

        return dataparser_outputs
