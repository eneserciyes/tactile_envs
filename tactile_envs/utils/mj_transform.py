import numpy as np
import open3d as o3d

def get_inverse_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(T.shape[0])
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def transform(X, T):
    """
    X: np.ndarray - (n, 3)
    T: transform - (4, 4)
    """
    return np.dot(T[:3, :3], X.T).T + T[:3, 3]

def quat_to_mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # This function is lifted directly from scipy source code
    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), \
        2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), \
        2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2]
    np_rot_mat = rot_list_to_mat(rot_mat_arr)
    return np_rot_mat

def rot_list_to_mat(rot_mat_arr):
    '''
    Generates numpy rotation matrix from rotation matrix as list len(9)
    @param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)
    @return np_rot_mat: 3x3 rotation matrix as numpy array
    '''
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat

def pos_rot_to_mat(pos, rot):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot
    t_mat[:3, 3] = np.array(pos)
    return t_mat

def cammat_to_o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def get_camera_projection_matrix(env, camera_name):
    ''' Returns camera pose. This is not extrinsics '''
    pos = env.data.cam_xpos[env.model.cam(camera_name).id]
    rot = env.data.cam_xmat[env.model.cam(camera_name).id].reshape((3,3))

    projection_matrix = pos_rot_to_mat(pos, rot)
    return projection_matrix

def get_robot_base_projection(env):
    pos = env.data.body('base').xpos
    rot = env.data.body('base').xmat.reshape((3,3))
    projection_matrix = pos_rot_to_mat(pos, rot)
    return projection_matrix

def get_robot_leftpad_dist(env):
    world_to_gripper = get_inverse_T(get_robot_gripper_projection(env))
    pos = env.data.body('leftpad').xpos
    rot = env.data.body('leftpad').xmat.reshape((3,3))
    leftpad_to_world = pos_rot_to_mat(pos, rot)
    projection_matrix = np.dot(leftpad_to_world, world_to_gripper)
    return projection_matrix[1, 3].item()

def get_robot_rightpad_dist(env):
    world_to_gripper = get_inverse_T(get_robot_gripper_projection(env))
    pos = env.data.body('rightpad').xpos
    rot = env.data.body('rightpad').xmat.reshape((3,3))
    rightpad_to_world = pos_rot_to_mat(pos, rot)
    projection_matrix = np.dot(rightpad_to_world, world_to_gripper)
    return projection_matrix[1, 3].item()

def get_robot_gripper_projection(env):
    pos = env.data.body('hand').xpos
    rot = env.data.body('hand').xmat.reshape((3,3))
    projection_matrix = pos_rot_to_mat(pos, rot)
    return projection_matrix

def get_camera_intrinsics(env, camera_name, img_size=None):
    """
    Link: https://github.com/openai/mujoco-py/issues/271
    """
    if img_size == None:
         img_size = env.height
    fovy = env.model.cam_fovy[env.model.cam(camera_name).id]
    fovy_radians = np.radians(fovy)

    f = 0.5 * img_size / np.tan(fovy_radians/ 2)
    camera_intrinsics = np.array([[f, 0, img_size / 2], [0, f, img_size / 2], [0, 0, 1]])
    return camera_intrinsics

def get_all_camera_transforms(env, camera_names):
    return {
         cam: get_camera_projection_matrix(env, cam)
         for cam in camera_names
    }

def get_all_camera_intrinsics(env, camera_names, img_size=None):
	return {cam: get_camera_intrinsics(env, cam, img_size) for cam in camera_names}
