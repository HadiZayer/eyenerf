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
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import rawpy
from PIL import Image
from torch.utils.data import Dataset
from jaxtyping import Float, UInt8
# from torchtyping import TensorType
import tifffile

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.has_masks = dataparser_outputs.mask_filenames is not None
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        # print(f"loading image {image_idx}")
        if "png" in str(image_filename):
            pil_image = Image.open(image_filename)
            if self.scale_factor != 1.0:
                width, height = pil_image.size
                newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
                pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
            image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        elif "ARW" in str(image_filename):
            params = rawpy.Params(use_camera_wb=True, bright=2.5)
            with rawpy.imread(str(image_filename)) as raw:
                image = raw.postprocess(params)
        elif "tif" in str(image_filename):
            image_uint = tifffile.imread(image_filename)
            if image_uint.max() > 2 ** 8:
                image = image_uint / (2**16 - 1)
            else:
                image = image_uint / (2**8 - 1)
        else:
            pil_image = Image.open(image_filename + ".png")
            if self.scale_factor != 1.0:
                width, height = pil_image.size
                newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
                pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
            image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)            
            # print(image_uint.max())
            
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        # assert image.dtype == np.uint8
        # print(image.max())
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        # print(f"finished loading image {image_idx}")
        return image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_numpy_image(image_idx) 
        if image.dtype == np.uint8:
            image = torch.from_numpy(image.astype("float32") / 255.0)
        else:
            image = torch.from_numpy(image).float()
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = image
        if self.has_masks:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        # metadata = self.get_metadata(data)
        if len(self.metadata["ellipse_params"]) > 0:
            metadata = {"ellipse_params": self.metadata["ellipse_params"][image_idx]}
        else:
            metadata = {"ellipse_params": []}
        data.update(metadata)
        return data

    # pylint: disable=no-self-use
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames
