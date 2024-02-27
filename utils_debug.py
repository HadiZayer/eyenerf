import plotly.graph_objects as go
import numpy as np

def plot_vectors(pts, vectors, skip_step, lens, fig, mask=None, colors=("red", "green")):
    pts = pts.detach().cpu().numpy()
    vectors = vectors.detach().cpu().numpy()
    lens = lens.detach().cpu().numpy()
    colorscale = [
        [0, colors[0]],
        [1.0, colors[1]],
    ]  # red is origin, green is point inwards wrt ray direction
    res = len(pts)
    for i in range(0, res, skip_step):
        if lens[i] == 0 or (mask is not None and mask[i] == 0):
            continue
        curr_vec = vectors[i]
        # curr_vec = curr_vec / np.linalg.norm(curr_vec)
        curr_pt = pts[i]
        ray_pt = curr_pt + curr_vec * lens[i]
        end_pts = np.stack([curr_pt, ray_pt], axis=0)
        curr_ray = go.Scatter3d(
            x=end_pts[:, 0],
            y=end_pts[:, 1],
            z=end_pts[:, 2],
            marker=dict(
                size=4,
                color=[0.0, 1.0],
                colorscale=colorscale,
            ),
            line=dict(color="darkblue", width=2),
        )
        fig.add_trace(curr_ray)

        