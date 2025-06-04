# This is the main loop for the One short Dual Arm Imitation Learning paper

import time as time_lib
from typing import Dict

import numpy as np
from gymnasium.spaces import Box

from irl_control.mujoco_gym_app import MujocoGymAppHighFidelity
from irl_control.utils.target import Target
import cv2
import numpy as np
import torch
import math

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from lightglue import match_pair

def get_camera_intrinsics_from_fovy(sim, img_width, img_height):
    cam_id = 1
    fovy_rad = sim.model.cam_fovy[cam_id] * np.pi / 180  # convert to radians
    fy = img_height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # assume square pixels
    cx = img_width / 2
    cy = img_height / 2
    return fx, fy, cx, cy

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

def mujoco_depth_to_real(depth, near=0.02, far=10.0):
    """
    Convert MuJoCo normalized depth to real-world depth in meters.
    """
    return  near * far / (far - (far - near) * depth)

def segment(image):
    """
    Make it Grounding DINO for segmentation of any object
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define broad range for blue (light to dark, cyan to navy)
    lower_blue = np.array([90, 40, 40])     # allow dull and light blue
    upper_blue = np.array([130, 255, 255])  # include strong and dark blues

    # Create mask
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to original image
    segmented_blue = cv2.bitwise_and(image, image, mask=blue_mask)
    return blue_mask


def find_center(segmented_mask):
    """
    Replace this function with PPCR for better prediction of the pose of the object.
    """
    moments = cv2.moments(segmented_mask)
    if moments["m00"] == 0:
        print("Mask has zero area")
        exit()

    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    print(f"Centroid of the segmented object in pixel coordinates: ({cX}, {cY})")

    return cX, cY

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
    
    return camera_coords, world_coords

def create_transform_matrix(position, euler_angles):
    """
    Create a transformation matrix from position and Euler angles.
    :param position: 3D position as a numpy array [x, y, z]
    :param euler_angles: Euler angles as a numpy array [roll, pitch, yaw]
    :return: 4x4 transformation matrix
    """
    roll, pitch, yaw = euler_angles

    # Create rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = position

    return transform_matrix

def feature_matching_algo(img0_np, img1_np):

    image0 = numpy_to_torch(img0_np).cuda()   # now shape = (3, H, W), float32, on GPU
    image1 = numpy_to_torch(img1_np).cuda()

    # ---- Initialize extractor and matcher ----
    extractor = SIFT(max_num_keypoints=2048).eval().cuda()
    matcher = LightGlue(features='disk').eval().cuda()

    # —– Get keypoint features and matches
    feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
    matches0 = matches01["matches0"].cpu().numpy()      # shape = (N0,)
    kpts0_all = feats0["keypoints"].cpu().numpy()      # shape = (N0, 2)
    kpts1_all = feats1["keypoints"].cpu().numpy()      # shape = (N1, 2)

    # —– Gather only the valid correspondences
    valid_mask = matches0 > -1
    pts0 = kpts0_all[valid_mask]                       # shape = (M, 2)
    pts1 = kpts1_all[matches0[valid_mask]]              # shape = (M, 2)

    print(f"Found {pts0.shape[0]} total matches.")

    # —– Check if there are ≥ 4 matches for a homography
    if pts0.shape[0] < 4:
        print("Not enough matches to compute a homography.")
        # You know “image0 is probably not in image1” or just skip H.
    else:
        # Convert to float32 (required by OpenCV)
        pts0_f = pts0.astype(np.float32)
        pts1_f = pts1.astype(np.float32)

        # Run RANSAC + Homography
        H, inlier_mask = cv2.findHomography(
            pts0_f, pts1_f,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995
        )

        if H is None:
            print("findHomography returned None → insufficient inliers.")
        else:
            num_inliers = int(inlier_mask.sum())
            print(f"Homography computed with {num_inliers}/{pts0.shape[0]} inliers.")
            # H is your 3×3 projective transform, mapping image0 → image1.

            # (Optional) Extract a pure rotation+translation via affine on the inliers:
            inlier_pts0 = pts0[inlier_mask.ravel().astype(bool)]
            inlier_pts1 = pts1[inlier_mask.ravel().astype(bool)]
            A, aff_mask = cv2.estimateAffinePartial2D(
                inlier_pts0.astype(np.float32),
                inlier_pts1.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
            if A is not None:
                r11, r12, tx = A[0]
                r21, r22, ty = A[1]
                theta = np.degrees(np.arctan2(r21, r11))
                print(f"2D‐Affine: rotation {theta:.2f}°, translation (tx={tx:.1f}, ty={ty:.1f})")
            else:
                print("Could not compute 2D‐affine from inliers.")


# 2. Convert BGR → RGB, then to float in [0,1], then to torch.Tensor, channel-first:
def numpy_to_torch(img_bgr: np.ndarray) -> torch.Tensor:
    # a) BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)      # still shape (H, W, 3)
    # b) uint8 [0..255] → float32 [0..1]
    img_f = img_rgb.astype(np.float32) / 255.0               # shape (H, W, 3)
    # c) H×W×3 → 3×H×W
    img_chw = np.transpose(img_f, (2, 0, 1))                 # shape (3, H, W)
    # d) to torch.Tensor
    tensor = torch.from_numpy(img_chw)                       # torch.float32 by default
    return tensor

    
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
        """
        Main loop for the robot control
        """
        # Camera Properties
        zfar = 5.0
        znear = 0.01
        fx, fy, cx, cy = get_camera_intrinsics_from_fovy(self, 480, 360)
        print(f"Model extent: {self.model.stat.extent}")
        print(f"Actual near: {znear * self.model.stat.extent}")
        print(f"Actual far: {zfar * self.model.stat.extent}")
        
        depth = self.mujoco_renderer.render("depth_array",camera_name="global_cam" )
        pixels = self.mujoco_renderer.render("rgb_array",camera_name="global_cam" )
        
        # Covert to actual depth
        real_depth = mujoco_depth_to_real(depth)

        ############## STAGE 1: Global Camera ##################
        # Get the segment from the global camera
        #demontration_mask = cv2.imread("assets/segmented_box.png", cv2.IMREAD_GRAYSCALE)
        demo_distance = 0.614
        demo_cX, demo_cY = 239, 309

        test_mask = segment(pixels)

        # Now from the segment find the mid Point of the segment
        test_cX, test_cY = find_center(test_mask)

        # Using the mid point, find the distance from the camera to the point
        center_depth = real_depth[test_cY, test_cX]
        print(f"Depth at center: {center_depth:.3f} meters")

        # Calculate the bottle neck position in the global camera frame using the current object pose
        # Usage example:
        global_camera_pos = [0, 0.8, -0.5]
        global_camera_euler = [-1.2, 0, 3.1415926536]
        img_width, img_height = 480, 360

        demo_camera_coords, demo_world_point = simple_pixel_to_world(
            demo_cX, demo_cY, demo_distance, 
            global_camera_pos, global_camera_euler, 60, 
            img_width, img_height
        )

        test_camera_coords, test_world_point = simple_pixel_to_world(
            test_cX, test_cY, center_depth, 
            global_camera_pos, global_camera_euler, 60, 
            img_width, img_height
        )

        test_world_point[2] += 0.04
        demo_world_point[2] += 0.04 
        print(f"Demo World coordinates: {demo_world_point}")
        print(f"Test World coordinates: {test_world_point}")

        # Finding all the transformation matrices for the estimate

        bottle_neck_pos = np.array([[0, 0.3, 0.25]])

        T_bottle_neck = create_transform_matrix(
            position=bottle_neck_pos[0],
            euler_angles=np.array([0, -np.pi / 2, 0])
        )
            
        T_demo_object = create_transform_matrix(
            position=demo_camera_coords,
            euler_angles=np.array([0, 0, 0])  # Assuming no rotation for the demo object
        )
        T_test_object = create_transform_matrix(
            position=test_camera_coords,
            euler_angles=np.array([0, 0, 0])  # Assuming no rotation for the test object
        )
        T_delta_global_camera = T_test_object @ np.linalg.inv(T_demo_object)
        print(f"Transformation from demo to test object:\n{T_delta_global_camera}")

        T_global_camera_robot_frame = create_transform_matrix(
            position=global_camera_pos,
            euler_angles=global_camera_euler
        )

        estimate = (((T_global_camera_robot_frame @ T_delta_global_camera) @ np.linalg.inv(T_global_camera_robot_frame)) @ T_bottle_neck)[:3, 3]
        print(f"Estimated bottle neck position in robot frame: {estimate}")

        # Define targets for both arms
        targets: Dict[str, Target] = {
            "base": Target(),
            "ur5right": Target(),
            "ur5left": Target(),
        }

        right_wp = np.array([estimate[0], estimate[1], estimate[2]])
        left_wp = np.array([-0.3, 0.45, 0.5])

        start_time = time_lib.time()
        ################## END OF STAGE 1 ##################

        # Main loop to control the robot
        while time_lib.time() - start_time < 10:

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

            # Render the cameras
            pixels = self.mujoco_renderer.render("rgb_array",camera_name="wrist_cam_left")
            cv2.imshow("Left Wrist Camera", pixels)
            depthr = self.mujoco_renderer.render("depth_array",camera_name="wrist_cam_right" )
            pixelsr = self.mujoco_renderer.render("rgb_array",camera_name="wrist_cam_right")
            cv2.imshow("Right Wrist Camera", pixelsr)
            depth = self.mujoco_renderer.render("depth_array",camera_name="global_cam" )
            pixels = self.mujoco_renderer.render("rgb_array",camera_name="global_cam" )
            cv2.imshow("Global Camera", pixels)
            cv2.waitKey(1)

        #################Stage 2: Wrist Camera#################################3
        # Get the segment from the wrist camera

        current_robot_pose = right_wp
        current_robot_orie = np.array([0, -np.pi / 2, 0])

        wrist_bottkleneck = cv2.imread("assets/expert_wrist_cam.jpeg")
        # open the numpy file 
        depthr_data = np.load("assets/depth_right.npy")
        demo_depthr_data = mujoco_depth_to_real(depthr_data)

        demo_wrist_mask = segment(wrist_bottkleneck)
        test_wrist_mask = segment(pixelsr)

        demo_wrist_cX, demo_wrist_cY = find_center(demo_wrist_mask)
        test_wrist_cX, test_wrist_cY = find_center(test_wrist_mask)

        # Calculate the distance from the camera to the wrist point
        demo_wrist_distance = demo_depthr_data[demo_wrist_cY, demo_wrist_cX]
        real_depthr = mujoco_depth_to_real(depthr_data)
        test_wrist_distance = real_depthr[test_wrist_cY, test_wrist_cX]

        print(f"Demo Wrist Depth: {demo_wrist_distance:.3f} meters")
        print(f"Test Wrist Depth: {test_wrist_distance:.3f} meters")

        # Do Light Glue
        feature_matching_algo(pixelsr, pixelsr)

        # Tranformation Matrixes for estimate

        T_EE_robot_frame = create_transform_matrix(
            position=current_robot_pose,
            euler_angles=current_robot_orie
        )

        wrist_cam_pose_EE = np.array([-0.05, 0, 0.05])
        wrist_cam_orie_EE = np.array([0, 1.56, 1.57079632679])

        T_wrist_cam_EE = create_transform_matrix(
            position=wrist_cam_pose_EE,
            euler_angles=wrist_cam_orie_EE
        )


        

        



if __name__ == "__main__":
    ur5 = MoveTest(robot_config_file="move_to_point.yaml", scene_file="kinect_environment.xml")
    ur5.run()
