import time as time_lib
from typing import Dict

import numpy as np
from gymnasium.spaces import Box

from irl_control.mujoco_gym_app import MujocoGymAppHighFidelity
from irl_control.utils.target import Target
import cv2
import numpy as np

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

        start_time = time_lib.time()
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
            pixels = self.mujoco_renderer.render("depth_array",camera_name="global_cam" )
            pixels = depth_to_rgb(pixels)
            cv2.imshow("Global Camera", pixels)
            cv2.waitKey(1)


if __name__ == "__main__":
    ur5 = MoveTest(robot_config_file="move_to_point.yaml", scene_file="kinect_environment.xml")
    ur5.run()


# 1.4 0 3.1415926536