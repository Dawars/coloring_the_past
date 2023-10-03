from pathlib import Path

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from nerfstudio.data.utils.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary

root = Path("/home/dawars/personal_projects/sdfstudio/data/heritage/")

for scene in tqdm(["theater_internal", "semperoper", "hotel_international", "observatory", "stockholm", "st_michael"]):
    data = root / scene
    image_dir = data / "dense/images"

    cams = read_cameras_binary(data / "dense/sparse/cameras.bin")
    imgs = read_images_binary(data / "dense/sparse/images.bin")
    pts3d = read_points3d_binary(data / "dense/sparse/points3D.bin")

    vis_dir = data / "pcd_reproj"
    vis_dir.mkdir(exist_ok=True)

    # key point depth
    pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4)
    error_array = torch.ones(max(pts3d.keys()) + 1, 1)
    for pts_id, pts in pts3d.items():
        pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz)
        error_array[pts_id, 0] = torch.from_numpy(pts.error)

    for img_id, img in imgs.items():
        # if img.name != "fortepan_19121.jpg":
        # if img.name != "fortepan_41842.jpg":
        #     continue
        camera = cams[img.camera_id]
        image = Image.open(image_dir / img.name).convert("RGB")

        # plt.imshow(image)
        # plt.show()

        # load sparse 3d points for each view
        # visualize pts3d for each image
        valid_3d_mask = img.point3D_ids != -1
        point3d_ids = img.point3D_ids[valid_3d_mask]
        img_p3d = pts3d_array[point3d_ids]
        img_err = error_array[point3d_ids]
        # img_p3d = img_p3d[img_err[:, 0] < torch.median(img_err)]

        # weight term as in NeuralRecon-W
        err_mean = img_err.mean()
        weight = 2 * np.exp(-((img_err / err_mean) ** 2))
        img_p3d[:, 3:] = weight

        rot = img.qvec2rotmat()
        tr = img.tvec.reshape(3, 1)
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
        height = camera.height
        width = camera.width

        # rot[:, 0] *= -1
        # rot[..., 1:3] *= -1  # flip y,z converting to nerfstudio/opengl camera tr

        pts_cam = img_p3d[:, :3] @ rot.T + tr[:, 0]

        # pts_cam[:, 1:3] *= -1
        # plt.scatter(*(pts_cam[:, :2]).T, s=1, c=pts_cam[:, 2])
        # plt.title("cam")
        # plt.show()

        pts2d_proj = pts_cam / pts_cam[:, 2:]

        proj = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )
        pts2d = pts2d_proj @ proj.T  # [0, 1, 2]

        # filter = (pts2d[:, 0] < width) & (pts2d[:, 0] >= 0) & (pts2d[:, 1] < height) & (pts2d[:, 1] >= 0)
        # pts2d = pts2d[filter]
        fig = plt.figure(frameon=False)
        # fig.set_size_inches(width, height)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.scatter(*(pts2d[:, :2]).T, s=1, c=pts_cam[:, 2])
        ax.imshow(image, aspect="auto")
        fig.savefig(vis_dir / img.name, dpi=100)
        plt.close()
        del fig
