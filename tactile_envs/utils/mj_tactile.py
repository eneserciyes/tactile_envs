import torch


MAX_TACT = torch.tensor([6.0, 6.0, 6.0, 0.06, 0.6, 0.006]).view(1, -1, 1, 1)


def preprocess_tactile_data(sensor_data):
    # TODO: probably need to cancel out forces in some directions, e.g. positive normal force
    tactile_data = (
        torch.from_numpy(sensor_data.copy()).reshape((2, 6, 2, 4)).permute(0, 1, 3, 2)
    )  # (num_sensors, num_channels, H, W)
    tactile_data_signs = torch.sign(tactile_data)
    tactile_data = torch.log1p(torch.abs(tactile_data).to(torch.float32))
    tactile_data = tactile_data * tactile_data_signs

    # normalize tactile data
    tactile_data = tactile_data / MAX_TACT
    # clamp the tactile data to [-1, 1]
    tactile_data = torch.clamp(tactile_data, -1, 1)
    return tactile_data


def normalize_tactile_data(tactile_data: torch.Tensor):
    """
    tactile_data: (T, S, C, H, W) - T: time, S: sensor, C: channels, H: height, W: width
    Returns: (T, S, C, H, W)
    """
    channel_max = tactile_data.amax(dim=(0, 1, 3, 4), keepdim=True)
    channel_min = tactile_data.amin(dim=(0, 1, 3, 4), keepdim=True)

    return (tactile_data - channel_min) / (channel_max - channel_min)


def taxel_to_3d(
    left_taxels, right_taxels, leftpad_dist, rightpad_dist, claw_height, claw_width
):
    """
    left_taxels, right_taxels: (C, H, W)
    leftpad_dist, rightpad_dist: float
    claw_height: float
    claw_width: float
    returns: ((3+C, H, W), (3+C, H, W))
    """
    C, H, W = left_taxels.shape

    # get a grid of coordinates
    y = torch.arange(H)
    x = torch.arange(W)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinate_grid = torch.stack([yy, xx], dim=-1)
    h = coordinate_grid[..., 0]
    w = coordinate_grid[..., 1]

    # compute the 3D coordinates of each taxel
    taxel_xz = torch.stack(
        (claw_height * (2 * h - 3) / 4, claw_width * (2 * w - 1) / 2), dim=-1
    )
    taxel_y = torch.ones((H, W, 1))
    taxel_xyz_left = torch.cat((taxel_y, taxel_xz), dim=-1)[..., [1, 0, 2]]

    # repeat the 3D coordinates for each sensor for T time steps
    taxel_xyz_left = taxel_xyz_left.permute(2, 0, 1)  # (3, H, W)

    # clone for right pad
    taxel_xyz_right = taxel_xyz_left.clone()

    # replace the y coordinate with the distance to the pad
    leftpad_dist = torch.tensor(leftpad_dist).reshape(-1, 1, 1).repeat(1, H, W)
    rightpad_dist = torch.tensor(rightpad_dist).reshape(-1, 1, 1).repeat(1, H, W)
    taxel_xyz_left[1:2] = leftpad_dist
    taxel_xyz_right[1:2] = rightpad_dist

    # add the tactile information to the 3D coordinates
    taxel_points_left = torch.cat((taxel_xyz_left, left_taxels), dim=0)
    taxel_points_right = torch.cat((taxel_xyz_right, right_taxels), dim=0)

    return (taxel_points_left, taxel_points_right)


def sample_tactile_points(tactile_centroids, n_sample=8, noise=0.0001):
    # tactile_centroids: (C, H, W)
    # n_sample: int
    # noise: float
    # returns: (n_sample*H*W, C)
    C, H, W = tactile_centroids.shape

    # generate position_noise
    position_noise = torch.randn((n_sample, 3, H, W)) * noise
    total_noise = torch.zeros((n_sample, C, H, W))
    total_noise[:, :3] = position_noise

    # sample tactile points
    tactile_pcd = tactile_centroids.unsqueeze(0) + total_noise

    # reshape the tactile points
    tactile_pcd = tactile_pcd.permute(0, 2, 3, 1)
    tactile_pcd = tactile_pcd.reshape(n_sample * H * W, C)

    return tactile_pcd
