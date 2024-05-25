import os
import matplotlib.pyplot as plt
import json
import numpy as np
import imageio
import sys
import rawpy
import torch
import cv2
import copy
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage import color
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation, binary_erosion
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import torchvision
from torchvision.ops import box_convert
import argparse 
import exiftool

parser = argparse.ArgumentParser() 
parser.add_argument("--data_path", type=str)
parser.add_argument("--mode", choices=["synthetic", "real"], default="real")
parser.add_argument("--sam_path", type=str, default="./sam_vit_h_4b8939.pth")
parser.add_argument("--grounding_dino_prefix", type=str, default="./GroundingDINO")
main_args = parser.parse_args()

data_path = main_args.data_path
labels_path = f"{data_path}_labels"
mask_dir = f"{data_path}/train_masks"
vis_dir = f"{data_path}/vis_ell"
dino_debug_dir = f"{data_path}/dino_debug"
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.makedirs(dino_debug_dir, exist_ok=True)

mode = main_args.mode
fake_tform = [
    [-1.0, 0, -0, 0.0],
    [0, 0, 1.0, 0.0],
    [0.0, 1.0, 0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if mode == "synthetic":
    edges_dir = f"{data_path}/edge_maps"
    # background_color = [193, 130, 1]
    threshold = 65
    os.makedirs(edges_dir, exist_ok=True)
    with open(os.path.join(data_path, f"transforms_train_start.json"), "r", encoding="UTF-8") as f:
        meta = json.load(f)
        frames = meta["frames"]
        for frame in frames:
            fp = frame["file_path"]
            name = fp.split("/")[-1]
            img = (plt.imread(fp + ".png") * 255).astype(np.uint8)[..., :3]

            # masking
            bg_color = img[0].mean(axis=0)  # extreme hack
            # print(bg_color, img[2999, 0])
            bg_color_np = bg_color[None, None, :]
            diff = np.array(img) - bg_color_np
            diff_norm = np.linalg.norm(diff, axis=-1)
            color_mask = diff_norm > threshold

            k = np.ones((3, 3), dtype=int)
            # k = np.zeros((3, 3), dtype=int)
            # k[1] = 1
            # k[:, 1] = 1  # for 8-connected
            color_mask_edges = binary_dilation(color_mask == 0, k) & color_mask

            imageio.imwrite(f"{edges_dir}/{name}.png", (color_mask_edges * 255).astype(np.uint8))
            # print(color_mask_edges[2999, 4999], (color_mask == 0)[2999, 4999])
            y, x = np.nonzero(color_mask_edges)
            pts = np.stack([y, x], axis=-1)
            dists = cdist(pts, pts)
            max_ind = np.argmax(dists)
            max_ind = np.unravel_index([max_ind], dists.shape)
            pt_1_ind, pt_2_ind = max_ind[0][0], max_ind[1][0]
            pt_1, pt_2 = pts[pt_1_ind], pts[pt_2_ind]
            print(pt_1, pt_2)

            ell_vis = img.copy()
            cv2.circle(ell_vis, (pt_1[1], pt_1[0]), 10, (255, 0, 0), -1)
            cv2.circle(ell_vis, (pt_2[1], pt_2[0]), 10, (255, 0, 0), -1)
            plt.imsave(f"{vis_dir}/{name}.png", ell_vis)

            mask_fp = f"{mask_dir}/{name}.png"
            k = np.ones((30, 3), dtype=int)  # for 4-connected
            color_mask = binary_dilation(color_mask == 1, k)
            color_mask = binary_erosion(color_mask, k)
            imageio.imwrite(mask_fp, (color_mask * 255).astype(np.uint8))
            start_transform = np.array(frame["transform_matrix"])
            frame["mask_path"] = mask_fp
            frame["transform_matrix"] = fake_tform
            print(start_transform[:3, 3])
            # break
    with open(os.path.join(data_path, f"transforms_train.json"), "w", encoding="UTF-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    train_t = start_transform[:3, 3]
    with open(os.path.join(data_path, f"transforms_test_start.json"), "r", encoding="UTF-8") as f:
        meta = json.load(f)
        frames = meta["frames"]
        for frame in frames:
            curr_tform = np.array(frame["transform_matrix"])
            curr_t = curr_tform[:3, 3]
            curr_tform[:3, 3] = curr_t - train_t
            frame["transform_matrix"] = listify_matrix(curr_tform)
    with open(os.path.join(data_path, f"transforms_test.json"), "w", encoding="UTF-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)


elif mode == "real":
    ellseg_path_2 = "ellseg"
    # sys.path.append(ellseg_path)
    sys.path.append(ellseg_path_2)

    from seg_utils import iris_segment
    from ellseg.modelSummary import model_dict
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
    from ellseg.evaluate_ellseg import make_parser
    from skimage.morphology import reconstruction

    parser = make_parser()
    args = parser.parse_args(["--save_maps", "0", "--ellseg_ellipses", "1"])

    loadfile = "ellseg/weights/all.git_ok"
    netDict = torch.load(loadfile)
    model = model_dict["ritnet_v3"]
    model.load_state_dict(netDict["state_dict"], strict=True)
    model = model.cuda()

    CHECKPOINT_PATH = main_args.sam_path
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "vit_h"

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)

    mask_predictor = SamPredictor(sam)
    filenames = os.listdir(data_path)
    filenames = [f for f in filenames if "tif" in f]
    out_dict = {}
    out_dict["frames"] = []
    for i, fn in enumerate(filenames):
        fp = f"{data_path}/{fn}"
        filename = fp.split("/")[-1]
        print('processing', filename)
        name = filename.split(".")[0]
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(fp)[0]

        # hacky (assumes parameters are the same across images)
        h, w = metadata["EXIF:ImageHeight"], metadata["EXIF:ImageWidth"]
        if "EXIF:FocalLength" in metadata:
            fl = metadata["EXIF:FocalLength"]
            print(fl)
            # out_dict["fl_x"] = fl * (w / 13.2)  # for sony camera
        else:
            fl = 0
        out_dict["image_width"] = w
        out_dict["image_height"] = h

        prefix = main_args.grounding_dino_prefix
        dino_model = load_model(
            f"{prefix}/groundingdino/config/GroundingDINO_SwinT_OGC.py", f"{prefix}/weights/groundingdino_swint_ogc.pth"
        )
        TEXT_PROMPT = "eye"
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25
        image_source, image = load_image(fp)
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        detections = Model.post_process_result(
            source_h=h,
            source_w=w,
            boxes=boxes,
            logits=logits)
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        plt.imsave(f"{dino_debug_dir}/{name}_dino.png", annotated_frame[..., ::-1])

        # https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/fe24c033820adffff66ac0eb191828542e8afe5e/grounded_sam_simple_demo.py#L61
        NMS_THRESHOLD = 0.5
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()
        nms_idx = sorted(nms_idx)
        print("nms_idx", nms_idx)
        xyxy_torch = torch.from_numpy(detections.xyxy)
        print(torchvision.ops.box_iou(xyxy_torch, xyxy_torch))

        detections.xyxy = detections.xyxy[nms_idx]
        # detections.confidence = detections.confidence[nms_idx]
        # detections.class_id = detections.class_id[nms_idx]

        boxes_old = boxes * torch.Tensor([w, h, w, h])
        xyxy_old = box_convert(boxes=boxes_old, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xyxy = detections.xyxy
        print(xyxy)
        print(xyxy_old)

        # masking
        crops = []

        if len(xyxy) != 2:
            print('enter crop manually')
            breakpoint()

        if xyxy[0][0] < xyxy[1][0]:
            order = ['right_eye', 'left_eye']
        else:
            order = ['left_eye', 'right_eye']

        for i, bbox in enumerate(xyxy):
            l, t, r, b = bbox.astype(int)
            print(l, t, r, b)

            crop_w = r - l
            crop_h = b - t
            ar = crop_w / crop_h
            target_ar = 320 / 240
            if ar < target_ar:
                target_crop_w = int(target_ar * crop_h)
                extra_w = (target_crop_w - crop_w) // 2
                l = l - extra_w
                r = r + extra_w
            elif ar > target_ar:
                target_crop_h = int(crop_w / target_ar)
                extra_h = (target_crop_h - crop_h) // 2
                t = t - extra_h
                b = b + extra_h
            else:
                pass
            crop = image_source[t:b, l:r, :]
            print(crop.shape)

            iris_ellipse, ell_vis, seg_map_disp, mask = iris_segment(model, sam, crop, args)

            y, x = np.nonzero(mask)
            pts = np.stack([y, x], axis=-1)
            dists = cdist(pts, pts)
            max_ind = np.argmax(dists)
            max_ind = np.unravel_index([max_ind], dists.shape)
            cx, cy, rx, ry, e = iris_ellipse
            print("fit rx: ", rx)
            max_dist = (dists[max_ind] / 2)[0]
            print("max_dist/2: ", max_dist)
            # if abs(max_dist - rx) > 2:
            #     continue 
            # rx = max_dist 
            # print('max x - min x')
            rx = (np.max(x) - np.min(x)) / 2.0
            print('max x - min x', rx)

            full_mask = np.zeros((h, w))
            full_mask[t:b, l:r] = mask

            print("eroding")
            seed = np.copy(full_mask)
            seed[1:-1, 1:-1] = full_mask.max()
            mask = full_mask

            filled = reconstruction(seed, mask, method="erosion")

            actual_cx = cx + l
            actual_cy = cy + t

            mask_fp = f"{mask_dir}/{name}_{i}.png"
            frame = {}
            frame["mask_path"] = mask_fp
            frame["ellipse_params"] = [actual_cx, actual_cy, rx, ry, e]
            frame["file_path"] = fp
            frame["transform_matrix"] = fake_tform
            frame["fl_x"] = fl * (w / 13.2)
            frame['order'] = order[i]
            out_dict["frames"].append(frame)
            imageio.imwrite(mask_fp, (filled * 255).astype(np.uint8))
            plt.imsave(f"{vis_dir}/{name}_{i}.png", ell_vis)
            plt.imsave(f"{vis_dir}/{name}_{i}_mask.png", (ell_vis * filled[t:b, l:r, None]).astype(np.uint8))
            plt.imsave(f"{vis_dir}/{name}_{i}_not_mask.png", (ell_vis * (1-filled[t:b, l:r, None])).astype(np.uint8))

    with open(os.path.join(data_path, f"transforms_train.json"), "w", encoding="UTF-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(data_path, f"transforms_test.json"), "w", encoding="UTF-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)