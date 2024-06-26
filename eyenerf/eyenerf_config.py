"""
EyeNerf configuration file.
"""

# from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from eyenerf.eyenerf_model import EyeNeRFModelConfig
from eyenerf.eyenerf_datamanager import EyenerfDataManagerConfig
from eyenerf.eyenerf_camera_optimizers import CameraOptimizerConfig
from eyenerf.eyenerf_dataparser import EyenerfDataParserConfig
from nerfstudio.configs.base_config import ViewerConfig
# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
# from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

from nerfstudio.plugins.types import MethodSpecification

# very hacky
cam_opt_opt_cfg = AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
cam_opt_sch_cfg = ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000)
eyenerf_method = MethodSpecification(
    TrainerConfig(
        method_name="cornea",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=EyenerfDataManagerConfig(
                dataparser=EyenerfDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="depth",
                    optimizer=cam_opt_opt_cfg,
                    scheduler=cam_opt_sch_cfg,
                ),
            ),
            model=EyeNeRFModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="radiance field from cornea reflections"
)