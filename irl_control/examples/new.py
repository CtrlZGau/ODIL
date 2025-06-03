import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

class MujocoWithCameras(MujocoEnv):
    def __init__(
        self,
        xml_path: str,
        frame_skip: int = 3,
        camera_names: list[str] = ("test_camera",),
        width: int = 480,
        height: int = 360,
        **kwargs,
    ):
        # Load model and data
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        super().__init__(model, data, frame_skip=frame_skip, **kwargs)

        # Store desired cams and resolution
        self.camera_names = camera_names
        self.width, self.height = width, height

        # Map camera names → their integer IDs in the XML
        self._cam_ids = {
            name: model.camera_name2id[name] for name in camera_names
        }

        # Create a single offscreen context we’ll re-use
        self._renderer = mujoco.Renderer(model, data, offscreen=True)

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        # add camera obs into info
        for name in self.camera_names:
            info[f"rgb_{name}"] = self.render_camera(name)
        return obs, reward, done, info

    def render_camera(self, name: str) -> np.ndarray:
        """
        Returns an (H, W, 3) uint8 RGB array from the named fixed camera.
        """
        cam_id = self._cam_ids[name]
        # configure the offscreen camera
        cam = self._renderer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_id

        # render into the offscreen buffer
        self._renderer.render(self.width, self.height)

        # grab the pixels
        img = self._renderer.read_pixels(self.width, self.height)
        # mujoco returns BGR by default—convert to RGB
        return img[:, :, ::-1]

    def render(self, mode="human"):
        return super().render(mode)

if __name__ == "__main__":
    # Example usage:
    env = MujocoWithCameras(
        xml_path="assets/kinect_environment.xml",
        camera_names=["test_camera", "left_eye", "right_eye"],
        width=640,
        height=480,
        frame_skip=3,
        render_mode="human",
    )
    obs, _ = env.reset()
    for _ in range(200):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        # info["rgb_test_camera"] is your numpy array from that camera
        if done:
            break
    env.close()
