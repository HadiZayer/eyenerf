from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig, DataManager, AnnotatedDataParserUnion
# from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from eyenerf.eyenerf_dataparser import EyenerfDataParserConfig

# from nerfstudio.data.dataparsers.cornea_dataparser import CorneaDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
# from nerfstudio.model_components.ray_generators import RayGenerator
from eyenerf.eyenerf_ray_generators import RayGenerator
# from nerfstudio.data.datasets.base_dataset import InputDataset
from eyenerf.eyenerf_base_dataset import InputDataset
# from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from eyenerf.eyenerf_camera_optimizers import CameraOptimizerConfig
from eyenerf.eyenerf_lie_groups import exp_map_SE3, exp_map_SO3xR3, so3_log_map
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.cameras.rays import RayBundle
from kornia.morphology import dilation
import nerfstudio.utils.poses as pose_utils

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torchvision
from pathlib import Path
from dataclasses import dataclass, field
import torch
from typing_extensions import Literal
from PIL import Image
# from nerfstudio.data.pixel_samplers import PixelSampler
from eyenerf.eyenerf_pixel_samplers import PixelSampler
import numpy as np
from torch.nn import Parameter
import cv2
import os 
from scipy.spatial.transform import Rotation as R_scipy

from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

import math
import pickle
import dataclasses


def unpack_pts_xyz(pts):
    return pts[..., 0], pts[..., 1], pts[..., 2]


def compute_cornea_normals(pts, e=0.5, R=7.8):
    xs, ys, zs = unpack_pts_xyz(pts)
    p = 1 - e**2

    grad_x = 2 * xs
    grad_y = 2 * ys
    grad_z = 2 * p * zs - 2 * R
    grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)
    grad = grad / torch.linalg.norm(grad, dim=-1, keepdims=True)

    return grad


def generate_rays_for_image_plane(camera_pose, f, h_res, w_res, h_fp, w_fp):
    """
    currently assumes all the intrinsic information is in f
    """
    R = camera_pose[:3, :3]  # TODO: use this properly
    t = camera_pose[:3, 3]

    # print(t)

    h_samp = torch.linspace(-h_fp / 2, h_fp / 2, h_res)
    w_samp = torch.linspace(-w_fp / 2, w_fp / 2, w_res)
    # hacks to do half pixel offset 
    h_samp = h_samp + (h_samp[1] - h_samp[0]) * 0.5 
    w_samp = w_samp + (w_samp[1] - w_samp[0]) * 0.5 
    h_grid, w_grid = torch.meshgrid(h_samp, w_samp, indexing="ij")
    z = torch.ones_like(h_grid) * (f)
    pts = torch.stack([w_grid, h_grid, z], dim=-1)
    # print(pts / f)

    vs = pts / torch.linalg.norm(pts, dim=-1, keepdims=True)
    os = torch.tile(t, (h_res, w_res, 1))
    return os, vs, pts/f


def batched_ray_ellipse_intersection(os, vs, e=0.5, R=7.8, t_max=2.18):
    """
    Args:
        os: (..., 3)
        vs: (..., 3)

    Returns:
        pts: (..., 3)
        ts: (..., 1)
    """
    os = os.clone()
    p = 1 - e**2
    vs_x, vs_y, vs_z = unpack_pts_xyz(vs)
    os_x, os_y, os_z = unpack_pts_xyz(os)

    a = vs_x**2 + vs_y**2 + p * vs_z**2
    b = 2 * (os_x * vs_x + os_y * vs_y + p * os_z * vs_z - R * vs_z)
    c = os_x**2 + os_y**2 + p * os_z**2 - 2 * R * os_z

    disc = b * b - 4 * a * c
    mask = disc > 0  # avoid gradient issues (can technically deal with t=0 case with more masking?)
    ts = (-b[mask] - torch.sqrt(disc[mask])) / (2 * a[mask])

    # mask out locations with no valid intersection

    # print(os.shape, ts.shape, vs.shape)
    pts = os[mask] + ts[..., None] * vs[mask]
    mask_z = pts[..., 2] < t_max
    mask_out = mask.clone()
    mask_out[mask] = mask[mask] & (mask_z)

    return pts[mask_z], ts[mask_z], mask_out

def batched_ray_plane_intersection(os, vs, t_max=2.18):
    """
    Args:
        os: (..., 3)
        vs: (..., 3)

    Returns:
        pts: (..., 3)
        ts: (..., 1)
    """
    os = os.clone()
    # print('os', os)
    vs_x, vs_y, vs_z = unpack_pts_xyz(vs)
    os_x, os_y, os_z = unpack_pts_xyz(os)

    # print(os.shape, ts.shape, vs.shape)
    ts = (os_z - t_max) / (-vs_z)
    # pts = torch.zeros_like(os)
    pts = os + vs * ts.unsqueeze(-1)

    return pts

def get_scene_data(
    color_mask,
    horizontal_sensor_size=36,
    focal_length=167,
    cx=None,
    cy=None,
    radius=None,
    radius_noise_level=0.0,
    radius_noise_clipping=0.0,
    radius_noise_level_fixed=0.0,
    index=None, # get rid of this later 
    hardcoded_xyz=False,
    hardcoded_xy=False,
    cx_noise_level=0.0,
    cx_noise_clipping=0.0,
    cy_noise_level=0.0,
    cy_noise_clipping=0.0,
    gt_rots=None,
    depth_scaling=1260,
    hardcoded_z=False,
    gt_locs=None,
    coordinate_convention="-y,-z,x",
):
    """
    Args:
        color_mask: (h, w)

    return:
        depth map (np array)
        normal map (np array) (r: maps x, g: maps y, b: maps z)
        mask for the cornea (boolean 2D array)
    """

    t_max = 2.18  # height: 0 to 2.18 mm
    e = 0.5  # eccentricity: 0.5
    R = 7.8  # radius of curvature at apex: 7.8 mm

    # we need the below code to save memory 
    k = torch.zeros((3, 3), dtype=torch.int)
    k[1] = 1
    k[:, 1] = 1  # for 8-connected
    d_in = (color_mask == 0)[None, None, ...]
    d_out = dilation(d_in, k)[0, 0]
    color_mask_edges = d_out & color_mask
    ys, xs = torch.nonzero(color_mask_edges, as_tuple=True)

    # ys, xs = torch.nonzero(color_mask, as_tuple=True)
    img_h, img_w = color_mask.shape

    # hacky way of checking for synthetic data
    # TODO: fix cx, cy calculation using cdist stuff
    if cx is None:
        cx = (
            torch.max(xs) + (torch.min(xs) - torch.max(xs)) // 2
        )  # how to prove this is invariant wrt ellipse rotation?
    if cy is None:
        cy = torch.max(ys) + (torch.min(ys) - torch.max(ys)) // 2
    if radius is None:
        pts = torch.stack([ys, xs], dim=-1)
        dists = torch.cdist(pts.float(), pts.float())
        # radius = (torch.max(xs) - torch.min(xs)) // 2
        radius = dists.max() / 2
        # if index is not None:
        #     radius = radius + index * 10
        # print(radius)

    vertical_sensor_size = (img_h / img_w) * horizontal_sensor_size
    # breakpoint()
    if hardcoded_xyz:

        # get cx,cy,r from gt x,y,z
        if coordinate_convention == "minus_y,-z,x":
            real_x = float(-gt_locs[index][1] * 1000)
            real_y = float(-gt_locs[index][2] * 1000)
            predicted_depth = float(gt_locs[index][0] * 1000)
        elif coordinate_convention == "y,-z,-x":
            real_x = float(gt_locs[index][1] * 1000)
            real_y = float(-gt_locs[index][2] * 1000)
            predicted_depth = float(-gt_locs[index][0] * 1000)
        elif coordinate_convention == "x,-z,y":
            real_x = float(gt_locs[index][0] * 1000)
            real_y = float(-gt_locs[index][2] * 1000)
            predicted_depth = float(gt_locs[index][1] * 1000)        

        cx = real_x * (focal_length / predicted_depth) * (img_w / horizontal_sensor_size) + (img_w/2.0)
        cy = real_y * (focal_length / predicted_depth) * (img_h / vertical_sensor_size) + (img_h/2.0)
        radius = 5.518562874251497 * (focal_length / predicted_depth) * (img_w / horizontal_sensor_size) 
        print(cx.item(), cy.item(), real_x, real_y, predicted_depth, radius)

    if radius_noise_level > 0.0:
        print("radius before:", radius)
        factor = np.clip(np.random.normal() * radius_noise_level, a_min=-radius_noise_clipping, a_max=radius_noise_clipping)
        delta_r = factor * radius
        radius += delta_r
        print("radius after:", radius)
        
    if radius_noise_level_fixed != 0.0:
        radius = radius + radius_noise_level_fixed * radius

    if cx_noise_level > 0.0:
        print("cx before:", cx)
        cx = cx.float()
        factor = np.clip(np.random.normal() * cx_noise_level, a_min=-cx_noise_clipping, a_max=cx_noise_clipping)
        delta_x = factor * cx
        cx += delta_x
        print("cx after:", cx)
        
    if cy_noise_level > 0.0:
        print("cy before:", cy)
        cy = cy.float()
        factor = np.clip(np.random.normal() * cy_noise_level, a_min=-cy_noise_clipping, a_max=cy_noise_clipping)
        delta_y = factor * cy
        cy += delta_y
        print("cy after:", cy)

    # x,y,z from cx,cy,r
    adjusted_radius = horizontal_sensor_size / img_w * radius
    predicted_depth = 5.518562874251497 / adjusted_radius * focal_length

    h_factor = (horizontal_sensor_size / img_w) * predicted_depth / focal_length
    v_factor = (vertical_sensor_size / img_h) * predicted_depth / focal_length

    real_x = (cx - img_w / 2.0) * h_factor
    real_y = (cy - img_h / 2.0) * v_factor
 
    cam_pose = torch.tensor(
        [[1.0, 0.0, 0.0, -real_x], [0.0, 1.0, 0.0, -real_y], [0.0, 0.0, 1.0, -predicted_depth], [0.0, 0.0, 0.0, 1.0]]
    )
    error = 0
    if radius_noise_level > 0:
        error += delta_r ** 2
    if cx_noise_level > 0:
        error += delta_x ** 2 
    if cy_noise_level > 0: 
        error += delta_y ** 2 
    error = math.sqrt(error)
    # TODO: can I generate the os and vs directly from the mask + nonzero function?
    os, vs, pts_n = generate_rays_for_image_plane(
        cam_pose, focal_length, img_h, img_w, vertical_sensor_size, horizontal_sensor_size
    )
    if gt_rots is not None:
        cmat = np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, -1.0, 0.0]]).astype(np.float32)
        euler_angles = gt_rots[index].astype(np.float32)
        rot = R_scipy.from_euler("XYZ", euler_angles, degrees=True)
        rot_mat = rot.as_matrix().T.astype(np.float32)
        rot_mat = cmat.T @ rot_mat @ cmat
        rot_mat = torch.from_numpy(rot_mat)

        os = (rot_mat @ os.reshape(-1, 3).T).T 
        vs = (rot_mat @ vs.reshape(-1, 3).T).T
        os = os.reshape(img_h, img_w, -1)
        vs = vs.reshape(img_h, img_w, -1)
    cv2.imwrite(f"imgs/{index}.png", color_mask.numpy().astype(np.uint8)*255)
    os_masked, vs_masked = os[color_mask>0], vs[color_mask>0]
    isect_pts, isect_ts, mask = batched_ray_ellipse_intersection(os_masked, vs_masked)
    isect_normals = compute_cornea_normals(isect_pts, e, R)

    if gt_rots is not None:
        isect_pts = (rot_mat.T @ isect_pts.T).T
        isect_normals = (rot_mat.T @ isect_normals.T).T
        # mask = mask.reshape(img_h, img_w)

    full_normals = torch.zeros((img_h, img_w, 3))
    full_depth = torch.zeros((img_h, img_w))

    composed_mask = color_mask.clone()
    composed_mask[color_mask] *= mask

    full_normals[composed_mask] = isect_normals
    full_depth[composed_mask] = isect_ts
    full_depth /= 1e3  # convert to meters!!

    composed_mask_numpy = composed_mask.cpu().numpy()


    kernel = np.ones((13, 13), np.uint8) # originally 13 by 13, rotation optimization is sensitive to this with hardcoded poses?
    # # using erode function on the input image to be eroded
    erodedimage = cv2.erode(composed_mask_numpy.astype(np.uint8), kernel, iterations=1)
    # erodedimage = composed_mask_numpy
    eroded_mask = torch.tensor(erodedimage) != 0

    extra_data = [cx, cy, radius, radius]
    return full_depth, full_normals, eroded_mask, extra_data, predicted_depth, error, real_x, real_y


@dataclass
class EyenerfDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: EyenerfDataManager)
    """Target class to instantiate."""
    dataparser: EyenerfDataParserConfig = EyenerfDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="depth")
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    radius_noise_level: float = 0.0
    """How much (additive white gaussian) noise is introduced to the radius estimate (to make the pose noisier)"""
    radius_noise_clipping: float = 0.0
    """How much we clip the noise to ensure no invalid radius values (like negative)"""
    cx_noise_level: float = 0.0
    """How much (additive white gaussian) noise is introduced to the radius estimate (to make the pose noisier)"""
    cx_noise_clipping: float = 0.0
    """How much we clip the noise to ensure no invalid radius values (like negative)"""
    cy_noise_level: float = 0.0
    """How much (additive white gaussian) noise is introduced to the radius estimate (to make the pose noisier)"""
    cy_noise_clipping: float = 0.0
    """How much we clip the noise to ensure no invalid radius values (like negative)"""
    radius_noise_level_fixed: float = 0.0
    """How much relative error is introduced to the radius estimate"""
    hardcoded_xyz: bool = False 
    """load in exact x, y, z for cornea position"""
    hardcoded_rotations: bool = False
    """load in exact cornea rotations"""
    optimize_cornea_rotation: bool = True
    """for ablation"""
    optimize_cornea_translation: bool = True 
    """for ablation"""
    gt_depth_loss: bool = False 
    """for debugging depth optimization"""
    recalculate_normals: bool = True
    """what it sounds like"""
    hardcoded_xy: bool = False 
    """exact x, y (camera coordinate system) only"""
    rotation_noise_level: float = 0.0
    """max value for rotation noise (uniform distribution), in degrees"""
    depth_scaling: float = 1260.0
    """factor in scaling depth for hardcoded xyz"""
    hardcoded_z: bool = False 
    """for ablation"""
    optimize_xy_only: bool = False 
    """for ablation"""
    optimize_z_only: bool = False 
    """for ablation"""
    coordinate_convention: str = "minus y,-z,x" # TODO: this is really hacky fix it later
    """transform scene cameras to colmap coordinate system"""


class EyenerfDataManager(DataManager):
    config: EyenerfDataManagerConfig
    train_dataparser_outputs: DataparserOutputs
    train_dataset: InputDataset
    eval_dataset: InputDataset
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: EyenerfDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()

        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = InputDataset(dataparser_outputs=self.train_dataparser_outputs, scale_factor=1.0)

        self.eval_dataset = InputDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="test"), scale_factor=1.0
        )
        super().__init__()

    # load images and compute geometry
    def setup_train(self):
        # hoist below init here for debugging
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataparser_outputs.cameras.size, device=self.device, rotation_noise_level=self.config.rotation_noise_level,
        )

        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=-1,
            num_times_to_repeat_images=-1,
            device=self.device,
            num_workers=self.world_size,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.geom_data = {
            "depth_maps": [],
            "normal_maps": [],
        }
        self.actual_masks = []
        self.orig_masks = []
        self.eye_centers = []  # N x y x x
        self.radii = []  # ry, rx
        self.initial_depths = []
        # this is in order? 
        # np.random.seed(42)
        self.pose_errors = []
        self.real_xs = []
        self.real_ys = []
        if self.config.hardcoded_rotations:
            rot_data_path = f"{str(self.dataparser.config.data)}/rotation_noises.pkl"
            with open(rot_data_path, "rb") as f:
                self.gt_rots = pickle.load(f)
            print(self.gt_rots)
        else:
            self.gt_rots = None
        if self.config.hardcoded_xy or self.config.hardcoded_xyz:
            loc_data_path = f"{str(self.dataparser.config.data)}/gt_locs.pkl"
            with open(loc_data_path, "rb") as f:
                self.gt_locs = pickle.load(f)
            print(self.gt_locs)
        else:
            self.gt_locs = None
        for i, train_data in enumerate(iter(self.train_dataset)):
            ellipse_params = train_data["ellipse_params"]
            mask = train_data["mask"]
            self.orig_masks.append(mask)

            if len(ellipse_params) > 0:
                hss = 13.2
                if len(ellipse_params) == 5:
                    cx, cy, rx, ry, _ = ellipse_params
                elif len(ellipse_params) == 4:
                    cx, cy, rx, ry = ellipse_params
            else: # synthetic data 
                hss = 36
                cx, cy, rx, ry = None, None, None, None
            fl = (
                self.train_dataparser_outputs.cameras.fx[i, 0] * hss / self.train_dataparser_outputs.cameras.width[i, 0]
            )
            isect_ts, isect_normals, mask, extra_data, predicted_depth, error, real_x, real_y = get_scene_data(
                        mask.bool()[..., 0],
                        horizontal_sensor_size=hss,
                        focal_length=fl,
                        cx=cx,
                        cy=cy,
                        radius=rx, # TODO remove this pls
                        radius_noise_level=self.config.radius_noise_level,
                        radius_noise_level_fixed=self.config.radius_noise_level_fixed,
                        radius_noise_clipping=self.config.radius_noise_clipping,
                        index=i,
                        hardcoded_xyz=self.config.hardcoded_xyz,
                        hardcoded_xy=self.config.hardcoded_xy,
                        cx_noise_level=self.config.cx_noise_level,
                        cx_noise_clipping=self.config.cx_noise_clipping,
                        cy_noise_level=self.config.cy_noise_level,
                        cy_noise_clipping=self.config.cy_noise_clipping,
                        gt_rots=self.gt_rots,
                        depth_scaling=self.config.depth_scaling,
                        hardcoded_z=self.config.hardcoded_z,
                        gt_locs=self.gt_locs,
                        coordinate_convention=self.config.coordinate_convention
            )
            self.geom_data["depth_maps"].append(isect_ts)
            self.geom_data["normal_maps"].append(isect_normals)
            self.actual_masks.append(mask)

            if cx is None:
                cx, cy, rx, ry = extra_data
            self.radii.append([ry, rx])
            self.eye_centers.append([cy, cx])
            self.initial_depths.append(predicted_depth)
            self.pose_errors.append(error)
            self.real_xs.append(real_x)
            self.real_ys.append(real_y)

        # self.radii = np.array(self.radii)
        # self.eye_centers = np.array(self.eye_centers)

        self.radii = torch.tensor(self.radii)
        self.eye_centers = torch.tensor(self.eye_centers).long().cuda()
        self.initial_depths = torch.tensor(self.initial_depths).cuda()
        self.real_xs = torch.tensor(self.real_xs).cuda()
        self.real_ys = torch.tensor(self.real_ys).cuda()
        # image_idx = self.train_image_dataloader.cached_collated_batch["image_idx"].cpu()
        self.geom_data["depth_maps"] = torch.stack(self.geom_data["depth_maps"]).cuda()
        self.geom_data["normal_maps"] = torch.stack(self.geom_data["normal_maps"]).cuda()
        self.actual_masks = torch.stack(self.actual_masks)
        self.orig_masks = torch.stack(self.orig_masks)
        # self.nonzero_indices = torch.nonzero(self.actual_masks[image_idx], as_tuple=False)
        self.nonzero_indices = torch.nonzero(self.actual_masks, as_tuple=False)
        c, y, x = torch.nonzero(self.actual_masks, as_tuple=True)

        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

        self.train_pixel_sampler = PixelSampler(num_rays_per_batch=self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(
            self.train_dataparser_outputs.cameras.to(self.device),
            self.train_camera_optimizer,
        )

        ray_bundle = self.train_ray_generator(self.nonzero_indices) # outputs world coords
        cornea_rays_o = ray_bundle.origins + self.geom_data["depth_maps"][c, y, x, None] * ray_bundle.directions
        self.cornea_rays_o = torch.zeros(list(self.geom_data["depth_maps"].shape) + [3]).cuda()
        self.cornea_rays_o[c, y, x] = cornea_rays_o.detach()

        img_inds = torch.arange(0, len(self.train_dataset))
        self.center_points = self.cornea_rays_o[img_inds, self.eye_centers[:, 0], self.eye_centers[:, 1]]
        self.center_rays_d = self.center_points / torch.linalg.norm(self.center_points, dim=-1, keepdims=True)

        # if self.config.gt_depth_loss:
        self.gt_depth_loss_tracking = []

    def setup_eval(self):
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=-1,
            num_times_to_repeat_images=-1,
            device=self.device,
            num_workers=self.world_size,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = PixelSampler(num_rays_per_batch=self.config.eval_num_rays_per_batch)
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataset.cameras.size, device=self.device, rotation_noise_level=0.0
        )
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.eval_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size,
        )

    # sample from stuff computed in setup
    def next_train(self, step, debug=False, get_all=False):
        """Returns the next batch of data from the train dataloader."""
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        if get_all:
            actual_mask = self.actual_masks[step:step+1]
            ray_indices = torch.nonzero(actual_mask, as_tuple=False) 
            # batch = ray_indices
            gt_vis = actual_mask[0][..., None] * self.train_dataset[step]["image"]
            ray_indices[:, 0] = step 
        else:
            image_batch["nonzero_indices"] = self.nonzero_indices
            batch = self.train_pixel_sampler.sample(image_batch)
            ray_indices = batch["indices"] 
        ray_bundle = self.train_ray_generator(ray_indices)  # does the offset/jitter matter?
        # image_idx_map = self.train_image_dataloader.cached_collated_batch["image_idx"].cpu()
        # actual_image_idx = image_idx_map[ray_indices[:, 0]]
        actual_image_idx = ray_indices[:, 0]
        depths = self.geom_data["depth_maps"][actual_image_idx, ray_indices[:, 1], ray_indices[:, 2]]

        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        if debug:
            old_rays_d = rays_d.clone()
            old_rays_o = rays_o.clone()

        if ray_bundle.metadata is None:
            ray_bundle.metadata = dict()

        new_rays_o = self.cornea_rays_o[ray_indices[:, 0], ray_indices[:, 1], ray_indices[:, 2]]
        # should I recalculate normals when pose optimizer is off? doesn't really matter
        # because normals won't change after initialization anyways
        if self.train_ray_generator.pose_optimizer.config.mode == "depth" or self.train_ray_generator.pose_optimizer.config.mode == "off":
            depth_delta = self.train_ray_generator.pose_optimizer.pose_adjustment # currently, this affects x, y, and z 
            x_delta = self.train_ray_generator.pose_optimizer.pose_adjustment_x
            y_delta = self.train_ray_generator.pose_optimizer.pose_adjustment_y

            if self.config.recalculate_normals:
                # recalculate rays_o + normals here
                img_width = self.train_dataparser_outputs.cameras.width[0, 0]
                img_height = self.train_dataparser_outputs.cameras.height[0, 0]
                fx = self.train_dataparser_outputs.cameras.fx[0, 0]
                fy = self.train_dataparser_outputs.cameras.fy[0, 0]
                initial_depths = self.initial_depths[ray_indices[:, 0]]
                z_offsets = initial_depths
                if self.config.optimize_cornea_translation and not self.config.optimize_xy_only:
                    z_offsets = z_offsets + depth_delta[ray_indices[:, 0]]*1e3
                factor_x = z_offsets / fx # convert to mm 
                factor_y = z_offsets / fy

                x_offsets = (self.eye_centers[ray_indices[:, 0], 1] - img_width / 2.0) * factor_x 
                y_offsets = (self.eye_centers[ray_indices[:, 0], 0] - img_height / 2.0) * factor_y
                if self.config.optimize_cornea_translation and not self.config.optimize_z_only:
                    x_offsets += x_delta[ray_indices[:, 0]]
                    y_offsets += y_delta[ray_indices[:, 0]]
                offsets = torch.stack([-x_offsets, -y_offsets, -z_offsets], dim=-1)

                rays_o_input = rays_o + offsets # rays_o is in cornea frame technically because it's just 0 
                rotation = exp_map_SO3xR3(self.train_ray_generator.pose_optimizer.rotation_component)[ray_indices[:, 0]][..., :3, :3]

                if not self.config.optimize_xy_only and not self.config.optimize_z_only:
                    rays_o_input = torch.sum(rays_o_input[..., None, :] * rotation.transpose(1, 2), dim=-1) # rotate in cornea frame 

                rays_d_input = torch.zeros_like(rays_d) # world frame 
                rays_d_input[:, 0] = -rays_d[:, 0]
                rays_d_input[:, 1] = -rays_d[:, 2]
                rays_d_input[:, 2] = -rays_d[:, 1] 

                if not self.config.optimize_xy_only and not self.config.optimize_z_only:
                    rays_d_input = torch.sum(rays_d_input[..., None, :] * rotation.transpose(1, 2), dim=-1) # rotate in cornea frame

                # is this masking useful? 
                isect_pts, _, mask = batched_ray_ellipse_intersection(rays_o_input, rays_d_input)
                # TODO add stuff here
                plane_isect_pts = batched_ray_plane_intersection(rays_o_input, rays_d_input, t_max=2.18)
                plane_isect_pts = plane_isect_pts[mask]
                # breakpoint()

                normals_recomputed = compute_cornea_normals(isect_pts) 
                if not self.config.optimize_xy_only and not self.config.optimize_z_only:
                    normals_recomputed = torch.sum(normals_recomputed[..., None, :] * rotation[mask], dim=-1) # rotate in cornea frame
                normals_fixed = torch.zeros_like(normals_recomputed) 
                normals_fixed[..., 0] = -normals_recomputed[..., 0]
                normals_fixed[..., 1] = -normals_recomputed[..., 2]
                normals_fixed[..., 2] = -normals_recomputed[..., 1]
                
                if not self.config.optimize_xy_only and not self.config.optimize_z_only:
                    isect_pts = torch.sum(isect_pts[..., None, :] * rotation[mask], dim=-1) # rotate in cornea frame 
                isect_pts = (isect_pts - offsets[mask]) / 1000
                new_rays_o = torch.zeros_like(isect_pts)
                new_rays_o[..., 0] = -isect_pts[..., 0]
                new_rays_o[..., 1] = -isect_pts[..., 2]
                new_rays_o[..., 2] = -isect_pts[..., 1]

                # mask other variables
                ray_bundle.pixel_area = ray_bundle.pixel_area[mask]
                ray_bundle.camera_indices = ray_bundle.camera_indices[mask]
                mask_cpu = mask.cpu()
                if not get_all:
                    batch["image"] = batch["image"][mask_cpu]
                    batch["mask"] = batch["mask"][mask_cpu]
                    batch["indices"] = batch["indices"][mask_cpu]
                    # breakpoint()
            else:
                depth_deltas = depth_delta[ray_indices[:, 0]].unsqueeze(-1)
                depth_delta_unit = self.center_rays_d[ray_indices[:, 0]]
                new_rays_o = new_rays_o + depth_deltas * depth_delta_unit

                # duplicated code? 
                normals = self.geom_data["normal_maps"][actual_image_idx, ray_indices[:, 1], ray_indices[:, 2]]
                normals_fixed = normals.clone() * 0
                normals_fixed[..., 0] = -normals[..., 0]
                normals_fixed[..., 1] = -normals[..., 2]
                normals_fixed[..., 2] = -normals[..., 1]

            if self.config.gt_depth_loss:
                ray_bundle.metadata["absolute_depth_error"] = new_rays_o[:, 1] + 1.2587
        else:
            # duplicated code? 
            normals = self.geom_data["normal_maps"][actual_image_idx, ray_indices[:, 1], ray_indices[:, 2]]
            normals_fixed = normals.clone() * 0
            normals_fixed[..., 0] = -normals[..., 0]
            normals_fixed[..., 1] = -normals[..., 2]
            normals_fixed[..., 2] = -normals[..., 1]
            mask = None

        new_rays_d = rays_d[mask] - 2 * torch.sum(rays_d[mask] * normals_fixed, dim=-1, keepdim=True) * normals_fixed

        ray_bundle.origins = new_rays_o
        ray_bundle.directions = new_rays_d

        eye_coord_delta = plane_isect_pts[:, :2]
        eye_coord_delta_frac = eye_coord_delta / 5.5

        if mask is not None:
            # ray_bundle.metadata["eye_coord"] = eye_coord_delta_frac[mask_cpu]
            ray_bundle.metadata["eye_coord"] = eye_coord_delta_frac # already masked 
            ray_bundle.metadata["directions_norm"] = ray_bundle.metadata["directions_norm"][mask]
            ray_bundle.metadata["left_eye_mask"] = self.train_dataparser_outputs.cameras.order[actual_image_idx][mask_cpu]
        else:
            ray_bundle.metadata["eye_coord"] = eye_coord_delta_frac
            ray_bundle.metadata["left_eye_mask"] = self.train_dataparser_outputs.cameras.order[actual_image_idx]

        self.train_count += 1
        if get_all:
            return ray_bundle, ray_indices, gt_vis, mask
        if debug:
            return (
                ray_bundle,
                batch,
                image_batch,
                self.geom_data,
                # normals,
                normals_fixed,
                old_rays_o,
                old_rays_d,
                depths,
                normals_recomputed,
                isect_pts
            )
        else:
            return ray_bundle, batch

    def next_eval(self, step):
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups
