import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops

from tactile_envs.utils.mj_transform import cammat_to_o3d, pos_rot_to_mat, quat_to_mat

CAM_TRANSFORMS = [
    np.array(
        [
            [8.52960338e-01, 3.52947755e-01, -3.84560195e-01, -1.19000000e-01],
            [-5.21975729e-01, 5.76751790e-01, -6.28409668e-01, -1.87000000e-01],
            [-2.77555756e-17, 7.36739611e-01, 6.76176564e-01, 2.74000000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [8.52960338e-01, -3.52947755e-01, 3.84560195e-01, 1.19000000e-01],
            [5.21975729e-01, 5.76751790e-01, -6.28409668e-01, -1.87000000e-01],
            [2.77555756e-17, 7.36739611e-01, 6.76176564e-01, 2.74000000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [-8.52960338e-01, 3.52947755e-01, -3.84560195e-01, -1.19000000e-01],
            [-5.21975729e-01, -5.76751790e-01, 6.28409668e-01, 1.87000000e-01],
            [2.77555756e-17, 7.36739611e-01, 6.76176564e-01, 2.74000000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [-8.52960338e-01, -3.52947755e-01, 3.84560195e-01, 1.19000000e-01],
            [5.21975729e-01, -5.76751790e-01, 6.28409668e-01, 1.87000000e-01],
            [-2.77555756e-17, 7.36739611e-01, 6.76176564e-01, 2.74000000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
]


def camera_intrinsics():
    IM_SIZE = 64
    FOVY = 45
    fovy_radians = np.radians(FOVY)
    f = 0.5 * IM_SIZE / np.tan(0.5 * fovy_radians)
    return np.array([[f, 0, IM_SIZE / 2], [0, f, IM_SIZE / 2], [0, 0, 1]])


CAM_MAT = camera_intrinsics()


def join_pcds(pcds):
    return {
        "points": np.concatenate([pcd["points"] for pcd in pcds.values()]),
        "colors": np.concatenate([pcd["colors"] for pcd in pcds.values()]),
    }


def generateCroppedPointCloudAndPoses(
    imgs,
    depths,
    bounds=None,
):
    """
    stride controls how to subsample the image point cloud.
    """
    cam2clouds = []
    cam_poses = []
    target_bounds = None

    img_size = imgs[0].shape[0]

    if bounds is not None:
        min_bound = bounds[0::2]
        max_bound = bounds[1::2]
        target_bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )

    for i in range(len(imgs)):
        color_img, depth_img = imgs[i], depths[i]

        # convert camera matrix and depth image to Open3D format, then generate point cloud
        od_cammat = cammat_to_o3d(CAM_MAT, img_size, img_size)
        od_depth = o3d.geometry.Image(depth_img)
        od_color = o3d.geometry.Image(np.ascontiguousarray(color_img))
        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            od_color, od_depth, depth_scale=1.0, convert_rgb_to_intensity=False
        )
        o3d_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image,
            od_cammat,
        )

        # Compute world to camera transformation matrix
        cam_T = CAM_TRANSFORMS[i]
        cam_pos = cam_T[:3, 3]
        c2b_r = cam_T[:3, :3]
        """In MuJoCo, we assume that a camera is specified in XML as a body with pose p, and that that body has a 
        camera sub-element with pos and euler 0.  Therefore, camera frame with body euler 0 must be rotated about
        x-axis by 180 degrees to align it with the world frame."""
        b2w_r = quat_to_mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = pos_rot_to_mat(cam_pos, c2w_r)
        o3d_cloud = o3d_cloud.transform(c2w)

        # If both minimum and maximum bounds are provided, crop cloud to fit inside them.
        if target_bounds is not None:
            o3d_cloud = o3d_cloud.crop(target_bounds)

        points = np.asarray(o3d_cloud.points)

        cam_poses.append(c2w)
        cam2clouds.append(points)

    joined_pcd = np.concatenate(cam2clouds, axis=0)

    # FPS subsampling
    _, sample_inds = torch3d_ops.sample_farthest_points(
        torch.from_numpy(joined_pcd).cuda().unsqueeze(0), K=1024
    )
    joined_pcd = joined_pcd[sample_inds.cpu().numpy()[0]]

    return joined_pcd
