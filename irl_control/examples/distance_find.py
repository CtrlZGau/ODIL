import time as time_lib
from typing import Dict

import numpy as np
from gymnasium.spaces import Box

from irl_control.mujoco_gym_app import MujocoGymAppHighFidelity
from irl_control.utils.target import Target
import cv2
import numpy as np
import open3d as o3d  # pip install open3d
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

def save_depth_array(depth_array, filename):
    """
    Save the MuJoCo depth array to a file for later use.
    Args:
        depth_array: numpy array containing depth data from MuJoCo
        filename: string path to save the depth data
    """
    # Save as a numpy binary file for exact data preservation
    np.save(filename, depth_array)
    return f"Depth array saved to {filename}.npy"


def get_camera_intrinsics_from_fovy(sim, img_width, img_height):
    cam_id = 1
    fovy_rad = sim.model.cam_fovy[cam_id] * np.pi / 180  # convert to radians
    fy = img_height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # assume square pixels
    cx = img_width / 2
    cy = img_height / 2
    return fx, fy, cx, cy

def create_point_cloud_from_depth_rgb(depth, rgb, fx, fy, cx, cy, near=0.01, far=5.0):
    H, W = depth.shape
    real_depth = near * far / (far - (far - near) * depth)

    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert pixel coordinates to camera frame
    x = (u - cx) * real_depth / fx
    y = (v - cy) * real_depth / fy
    z = real_depth

    # Stack into N x 3 point array
    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3) / 255.0  # normalize RGB

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb_flat)
    return pcd

def save_depth_16bit(depth_array, filename):
    # Convert to millimeters and save as 16-bit PNG
    # This preserves depth precision up to 65.535 meters with mm accuracy
    depth_mm = np.clip(depth_array * 1000, 0, 65535).astype(np.uint16)
    cv2.imwrite(filename, depth_mm)
    print(f"Saved 16-bit depth image: {filename}")
    return depth_mm

def depth_to_rgb(depth, normalize=True):
    if normalize:
        # Normalize depth to 0–255
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)
    else:
        depth_normalized = (depth * 255).astype(np.uint8)  # works if depth is already in 0–1 range

    # Apply colormap (e.g., JET)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colored

def get_distance_from_depth(depth_image, x, y, scale=1.0):
    """
    Get the distance at pixel (x, y) from a depth image.

    Parameters:
    - depth_image: 2D NumPy array of shape (H, W)
    - x, y: Pixel coordinates
    - scale: Conversion factor if depth is not in meters (e.g., 0.001 if in mm)

    Returns:
    - distance: Distance in meters (or scaled units)
    """
    h, w = depth_image.shape
    if 0 <= y < h and 0 <= x < w:
        raw_depth = depth_image[y, x]  # Note: (y, x), not (x, y)
        distance = raw_depth * scale
        return distance
    else:
        raise ValueError("Pixel coordinates out of bounds.")


def deproject_pixel_to_point(u, v, depth, fx, fy, cx, cy):
    z = depth  # depth should be in meters
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])

    

def mujoco_depth_to_real(depth, near=0.02, far=10.0):
    """
    Convert MuJoCo normalized depth to real-world depth in meters.
    """
    return near * far / (far - (far - near) * depth)


class MoveTest(MujocoGymAppHighFidelity):
    """
    This class implements the Admittance Controller on Dual UR5 robot
    """

    def __init__(self, robot_config_file: str = None, scene_file: str = None):
        observation_space = Box(low=-np.inf, high=np.inf)
        action_space = Box(low=-np.inf, high=np.inf)

        # Initialize the Parent class with the config file
        super().__init__(
            robot_config_file,
            scene_file,
            observation_space,
            action_space,
            osc_use_admittance=True,
            render_mode="rgb_array"
        )

    @property
    def default_start_pt(self):
        return None

    def run(self):
        # Define targets for both arms
        targets: Dict[str, Target] = {
            "base": Target(),
            "ur5right": Target(),
            "ur5left": Target(),
        }

        right_wp = np.array([0, 0.3, 0.25])
        left_wp = np.array([-0.3, 0.45, 0.5])
        x, y = 239, 309
        start_time = time_lib.time()
        # Get camera intrinsics
        fx, fy, cx, cy = get_camera_intrinsics_from_fovy(self, 480, 360)
        while time_lib.time() - start_time < 20:
            # Set the target position and orientation of both arms
            targets["ur5right"].set_xyz(right_wp)
            targets["ur5right"].set_abg(np.array([0, -np.pi / 2, 0]))

            targets["ur5left"].set_xyz(left_wp)
            targets["ur5left"].set_abg(np.array([0, -np.pi / 2, 0]))

            # Generate forces from the OSC
            ctrlr_output = self.controller.generate(targets)

            ctrl = np.zeros_like(self.data.ctrl)
            for force_idx, force in zip(*ctrlr_output):
                ctrl[force_idx] = force

            self.do_simulation(ctrl, self.frame_skip)

            pixels = self.mujoco_renderer.render("rgb_array",camera_name="wrist_cam_left")
            cv2.imshow("Left Wrist Camera", pixels)
            pixels = self.mujoco_renderer.render("rgb_array",camera_name="wrist_cam_right")
            depth_right = self.mujoco_renderer.render("depth_array",camera_name="wrist_cam_right" )
            cv2.imshow("Right Wrist Camera", pixels)
            depth = self.mujoco_renderer.render("depth_array",camera_name="global_cam" )
            pixels1 = self.mujoco_renderer.render("rgb_array",camera_name="global_cam" )
            depth = mujoco_depth_to_real(depth)
            distance = get_distance_from_depth(depth, x, y)
            print(f"Distance at pixel ({x}, {y}): {distance:.2f} meters")
            # depth = create_point_cloud_from_depth_rgb(
            #     depth, pixels, fx, fy, cx, cy
            # )
            # pixels = depth_to_rgb(pixels)
            cv2.imshow("Global Camera", depth)
            cv2.waitKey(1)

        depth_value = deproject_pixel_to_point(309, 239, distance, fx, fy, cx, cy)
        print(f"Deprojected point at pixel (309, 239): {depth_value}")

        # depth = mujoco_depth_to_real(depth)
        # depth_right = mujoco_depth_to_real(depth_right)

        cv2.imwrite("assets/expert_global_cam.jpeg", pixels1)
        cv2.imwrite("assets/expert_wrist_cam.jpeg", pixels)

        #o3d.visualization.draw_geometries([depth])
        saved_img = save_depth_array(depth, 'assets/depth_global')
        saved_img_right = save_depth_array(depth_right, 'assets/depth_right')
        # save_depth_16bit(depth, "depth_16bit.png")
        # save_depth_16bit(depth_right, "depthr_16bit.png")



if __name__ == "__main__":
    ur5 = MoveTest(robot_config_file="move_to_point.yaml", scene_file="kinect_environment.xml")
    ur5.run()