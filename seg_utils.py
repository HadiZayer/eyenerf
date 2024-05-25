import cv2
from ellseg.modelSummary import model_dict
from ellseg.evaluate_ellseg import preprocess_frame, evaluate_ellseg_on_image, rescale_to_original, make_parser
from skimage import draw
import numpy as np
from segment_anything import SamPredictor
import matplotlib.pyplot as plt

def mp_roi_unnorm(h, w, roi):
    roi_float = [float(x.strip()) for x in roi.split(" ")]
    x_center, y_center, width, height = roi_float 
    x_center, width = int(w * x_center), int(w * width) 
    y_center, height = int(h * y_center), int(h * height)
    return x_center, y_center, width, height    

def draw_rect_mp_roi(image, roi):
    h, w, _ = image.shape 
    x_center, y_center, width, height = mp_roi_unnorm(h, w, roi)
    top_left = (x_center-width//2, y_center+height//2)
    bottom_right = (x_center+width//2, y_center-height//2)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 3)
    return top_left, bottom_right

def crop_roi(image, roi):
    h, w, _ = image.shape 
    x_center, y_center, width, height = mp_roi_unnorm(h, w, roi)
    left = x_center - width//2
    right = x_center + width//2 
    bottom = y_center + height//2 
    top = y_center - height//2
    out = image[top:bottom, left:right, :]
    return out

def draw_lms_mp(image, lms, radius=5, thickness=2):
    h, w, _ = image.shape
    for lm in lms: 
        lm_float = [float(x.strip()) for x in lm.split(" ")]
        lm_x, lm_y = lm_float[0], lm_float[1] 
        center = (int(lm_x * w), int(lm_y * h))
        cv2.circle(image, center, radius, (0, 255, 0), thickness)

def get_bb(pts):
    pts_x = pts[..., 0]
    pts_y = pts[..., 1]
    y_min, y_max = pts_y.min(), pts_y.max()
    x_min, x_max = pts_x.min(), pts_x.max()
    target_height = int((x_max - x_min) * (240 / 320))
    y_mid = int((y_max + y_min) / 2)
    y_min = y_mid - target_height//2
    y_max = y_min + target_height
    return (slice(y_min, y_max), slice(x_min, x_max), slice(None, None, None))

def iris_segment(model, sam, eye_image, args):
    mask_predictor = SamPredictor(sam)
    eye_image_orig = eye_image.copy()
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)
    eye_image, scale_shift = preprocess_frame(eye_image, (240, 320))
    print(eye_image.shape)
    seg_map, latent, pupil_ellipse, iris_ellipse = evaluate_ellseg_on_image(eye_image.unsqueeze(0).cuda(), model, args)
    seg_map, pupil_ellipse, iris_ellipse = rescale_to_original(seg_map,
                                                                pupil_ellipse,
                                                                iris_ellipse,
                                                                scale_shift,
                                                                eye_image_orig.shape)
    c_x, c_y, r_x, r_y, _ = iris_ellipse
    nz_h, nz_w = np.nonzero(seg_map)
    ell_l, ell_r = nz_w.min(), nz_w.max() 
    ell_t, ell_b = nz_h.min(), nz_h.max()
    mask_predictor.set_image(eye_image_orig)
    masks, scores, logits = mask_predictor.predict(
        box=np.array((ell_l, ell_t, ell_r, ell_b)),
        multimask_output=True
    )
    seg_map_disp = np.tile(seg_map[..., None] / 2.0, (1, 1, 3))
    cv2.rectangle(seg_map_disp, (ell_l, ell_t), (ell_r, ell_b), (1, 0, 0), 10)
    ell_vis = eye_image_orig.copy()
    ell_bb_crop = eye_image_orig[ell_t-10:ell_b+10, ell_l-10:ell_r+10, :]
    draw_ellipse(ell_vis, iris_ellipse)
    ind = np.argmax(scores)
    return iris_ellipse, ell_vis, seg_map_disp, masks[ind]

# TODO: scale this properly? 
def draw_ellipse(img, ell_params):
   [rr_i, cc_i] = draw.ellipse_perimeter(int(ell_params[1]),
                                          int(ell_params[0]),
                                          int(ell_params[3]),
                                          int(ell_params[2]),
                                          orientation=ell_params[4])
   rr_i = rr_i.clip(6, img.shape[0]-6)
   cc_i = cc_i.clip(6, img.shape[1]-6)
   img[rr_i, cc_i, ...] = np.array([255, 0, 0])

