# The Radiance Field in an Eye
Our codebase is built on top of Nerfstudio 1.0.2. Our installation process will install Nerfstudio automatically and install our method as an extension of Nerfstudio 

## Environment Installation
- Ensure that CUDA is available. This code has been verified to work with CUDA 11.8. 
- Run `conda env create -f environment.yml -v` in your terminal to generate the initial environment. 
- Activate the environment i.e. run `conda activate eyenerf`.
- Install tinycudann by running 
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn.git@a77dc53ed770dd8ea6f78951d5febe175d0045e9#subdirectory=bindings/torch
```

- Additionally, Exiftool and Grounding DINO need to be installed. 
    - Instructions for installing Exiftool can be found here: https://exiftool.org/install.html. Our code is confirmed to work with ExifTool 12.77. Please make sure that the ExifTool binary can be found on your `PATH`, as our code depends on PyExifTool which needs to be able to find ExifTool. 
    - GroundingDINO should be installed from this commit: https://github.com/IDEA-Research/GroundingDINO/tree/39b1472457b8264adc8581d354bb1d1956ec7ee7. 
- Then run `pip install -e .` from within this repo's folder. 

## Run the method
To run our method on our dataset, download one of our captures (found here -- [google drive](https://drive.google.com/file/d/1YMHZUSifun5gA2sKHj8Jiu32dnqFDccr/view?usp=sharing)), and run the following command, replacing the variables where appropriate: 

```
ns-train cornea --data $PATH_TO_DATA --output_dir $PATH_TO_OUTPUT --experiment_name $EXPERIMENT_NAME --pipeline.model.rot-loss-mult 0.1 --pipeline.model.use-texture-field True
```
The output will be saved in the directory at `$EXPERIMENT_NAME`. The results can be visualized here when running here: http://localhost:7007.
If you are running the code on a cluster, then you will need to forward the port (see https://docs.nerf.studio/quickstart/viewer_quickstart.html). 

## Data capture instructions
To capture your own data, we suggest the following steps:
- The reconstruction setup assumes a static camera, and a moving person, so make sure that your camera is on a sturdy tripod. To avoid any motion blur, we also recommend using a self-timer to capture to avoid shaking the camera.
- Since the signals in the eye reflection is small, we suggest capturing RAW photos to avoid any compression.
- For ideal capturing, ensure that the reflection in the eye is in focus, and increase the aperture size as needed to get as much reflection light into the image, while taking into account the exposure and motion blur.
- As a sanity check, zoom-in on the eye in the captured image and make sure that there is a visible reflection in the eye, as that is what will be used in the reconstruction.
- For denoising the RAW images, we recommend using Adobe Lightroom AI Denoise feature, and export the images to a 16-bits TIFF format (rather than 8-bits) to minimize data loss

## Data processing instructions
- Once capture the photos are captured, they need to be processed with the postprocessing script, `gen_masks_offline.py`. 
- There are 3 command line arguments that need to be set: `--data_path`, `--sam_path`, and `--grounding_dino_prefix`. 
- `--data_path` should be set to a folder that contains image data, in TIF format. 
- `--sam_path` should be set to where the Segment Anything ViT backbone is (should look something like `sam_vit_h_4b8939.pth`). 
- `--grounding_dino_prefix` should be set to path to the local directory containing the Grounding DINO repo (a folder named `GroundingDINO`). 



 
