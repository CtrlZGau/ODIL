import time as time_lib
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box

import mujoco
from mujoco import mj_name2id, mjtObj, Renderer

from irl_control.mujoco_gym_app import MujocoGymAppHighFidelity
from irl_control.utils.target import Target


class MoveTest(MujocoGymAppHighFidelity):
    """
    This class implements the Admittance Controller on Dual UR5 robot
    with camera image rendering.
    """

    def __init__(self, robot_config_file: str = None, scene_file: str = None):
        observation_space = Box(low=-np.inf, high=np.inf)
        action_space = Box(low=-np.inf, high=np.inf)

        # Initialize the parent class with the config file
        super().__init__(
            robot_config_file,
            scene_file,
            observation_space,
            action_space,
            osc_use_admittance=True,
            render_mode="human",
        )

        self.model = mujoco.MjModel.from_xml_path('assets/kinect_environment.xml')
        data = mujoco.MjData(self.model)

        # Setup Mujoco renderer
        self.renderer = Renderer(self.model)

        # List of camera names as defined in your XML
        self.camera_names = ['wrist_cam_right', 'wrist_cam_left', 'global_cam']
        self.camera_ids = {
            cam: mj_name2id(self.model, mjtObj.mjOBJ_CAMERA, cam)
            for cam in self.camera_names
        }

    @property
    def default_start_pt(self):
        return None

    def render_cameras(self):
        """Render images from multiple cameras and return them as a dict."""
        images = {}
        for cam_name, cam_id in self.camera_ids.items():
            self.renderer.update_scene(self.data, camera=cam_id)
            img = self.renderer.render()
            images[cam_name] = img
        return images

    def display_images(self, images: Dict[str, np.ndarray]):
        """Display images using matplotlib."""
        for cam_name, img in images.items():
            plt.imshow(img)
            plt.title(cam_name)
            plt.axis('off')
            plt.show()

    def run(self):
        # Define targets for both arms
        targets: Dict[str, Target] = {
            "base": Target(),
            "ur5right": Target(),
            "ur5left": Target(),
        }

        right_wp = np.array([0.5, 0.45, 0.5])
        left_wp = np.array([-0.3, 0.45, 0.5])

        start_time = time_lib.time()
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

            # Render environment (if needed)
            self.render()

            # Capture and display camera images
            images = self.render_cameras()
            self.display_images(images)  # Optional: comment this out if too slow


if __name__ == "__main__":
    ur5 = MoveTest(
        robot_config_file="move_to_point.yaml",
        scene_file="kinect_environment.xml"
    )
    ur5.run()
