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
import math

def get_camera_intrinsics_from_fovy(sim, img_width, img_height):
    cam_id = 1
    fovy_rad = sim.model.cam_fovy[cam_id] * np.pi / 180  # convert to radians
    fy = img_height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # assume square pixels
    cx = img_width / 2
    cy = img_height / 2
    return fx, fy, cx, cy

import numpy as np
import math

def simple_pixel_to_world(u, v, depth_meters, camera_pos, camera_euler, fovy=60, img_width=640, img_height=480):
    """
    Corrected pixel-to-world conversion for MuJoCo.
    u = column (x-axis), v = row (y-axis)
    """
    
    # Step 1: Camera intrinsics
    fovy_rad = math.radians(fovy)
    fy = img_height / (2 * math.tan(fovy_rad / 2))
    fx = fy
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    # Step 2: Convert pixel to normalized camera coordinates
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    
    # Step 3: Scale by depth to get 3D camera coordinates
    x_cam = x_norm * depth_meters
    y_cam = y_norm * depth_meters
    z_cam = depth_meters
    
    camera_coords = np.array([x_cam, y_cam, z_cam])
    
    # Step 4: MuJoCo coordinate system correction
    # MuJoCo cameras look down -Z, but may need Y-flip
    camera_coords[1] = -camera_coords[1]  # Flip Y if needed
    
    # Step 5: Rotation matrices (same as your version)
    rx, ry, rz = camera_euler
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = Rz @ Ry @ Rx
    
    # Step 6: Transform to world coordinates
    world_coords = R @ camera_coords + np.array(camera_pos)
    
    return world_coords



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
    

def mujoco_depth_to_real(depth, near=0.02, far=10.0):
    """
    Convert MuJoCo normalized depth to real-world depth in meters.
    """
    return  near * far / (far - (far - near) * depth)


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
        x, y = 240, 180
        start_time = time_lib.time()
        # Get camera intrinsics
        fx, fy, cx, cy = get_camera_intrinsics_from_fovy(self, 480, 360)
        print(f"Model extent: {self.model.stat.extent}")
        print(f"Actual near: {0.01 * self.model.stat.extent}")
        print(f"Actual far: {5.0 * self.model.stat.extent}")

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
            cv2.imshow("Right Wrist Camera", pixels)
            depth = self.mujoco_renderer.render("depth_array",camera_name="global_cam" )
            pixels = self.mujoco_renderer.render("rgb_array",camera_name="global_cam" )
            # distance = get_distance_from_depth(pixels, x, y)
            # print(f"Distance at pixel ({x}, {y}): {distance:.2f} meters")
            #depth = create_point_cloud_from_depth_rgb(
            #    depth, pixels, fx, fy, cx, cy
            #)
            # pixels = depth_to_rgb(pixels)
            cv2.imshow("Global Camera", pixels)
            cv2.waitKey(1)
        #o3d.visualization.draw_geometries([depth])

        print("This is the depth at the pixel (239, 309):", depth[239, 309])

        znear = 0.01
        zfar = 5.0

        depth = mujoco_depth_to_real(depth)

        print("This is the depth at the pixel (239, 309):", depth[309, 239])

        segmented_mask = cv2.imread("binary_blue_mask.png", cv2.IMREAD_GRAYSCALE)
        mask_indices = np.argwhere(segmented_mask > 0)

        object_points = []

        for v, u in mask_indices:
            z = depth[v, u]
            if z == 0 or np.isnan(z):  # skip invalid depth
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            object_points.append([x, y, z])

        object_points = np.array(object_points) * 2

        if len(object_points) == 0:
            print("No valid depth points in segmented region.")
            exit()

        # 4. Optional: Filter out outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

        filtered_points = np.asarray(pcd.points)

        # 5. Compute centroid of the object
        centroid = np.mean(filtered_points, axis=0)

        print("Estimated 3D position of the object (camera frame):", centroid)

        # 6. (Optional) Save segmented point cloud
        o3d.io.write_point_cloud("object_segment.pcd", pcd)

        moments = cv2.moments(segmented_mask)
        if moments["m00"] == 0:
            print("Mask has zero area")
            exit()

        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        print(f"Centroid of the segmented object in pixel coordinates: ({cX}, {cY})")
        center_depth = depth[cY, cX]

        print(f"Depth at center: {center_depth:.3f} meters")

        # Usage example:
        camera_pos = [0, 0.8, -0.5]
        camera_euler = [-1.2, 0, 3.1415926536]
        pixel_u, pixel_v = 239, 309
        depth_at_pixel = depth[pixel_v,pixel_u]  # From your working depth conversion
        img_width, img_height = 480, 360

        world_point = simple_pixel_to_world(
            pixel_u, pixel_v, depth_at_pixel, 
            camera_pos, camera_euler, 60, 
            img_width, img_height
        )

        print(f"World coordinates: {world_point}")

           

if __name__ == "__main__":
    ur5 = MoveTest(robot_config_file="move_to_point.yaml", scene_file="kinect_environment.xml")
    ur5.run()