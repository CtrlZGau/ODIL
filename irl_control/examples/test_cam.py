import time

import mujoco
import mujoco.viewer
import cv2
import numpy as np

# After collecting all frames
output_filename = 'output_video.mp4'


model = mujoco.MjModel.from_xml_path('assets/kinect_environment.xml')
data = mujoco.MjData(model)

duration = 2.5  # (seconds)
framerate = 60  # (Hz)
height = 320
width = 640

# Simulate and display video.
frames = []
frames1 = []
frames2 = []
mujoco.mj_resetData(model, data)  # Reset state and time.
with mujoco.Renderer(model, height, width) as renderer:
  while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data, camera='wrist_cam_left')
      pixels = renderer.render()
      frames.append(pixels)
      renderer.update_scene(data, camera='wrist_cam_right')
      pixels = renderer.render()
      frames1.append(pixels)
      renderer.update_scene(data, camera='global_cam')
      pixels = renderer.render()
      frames2.append(pixels)
print(f"Captured {len(frames)} frames at {framerate} FPS.")

height, width, _ = frames[0].shape
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))
height, width, _ = frames1[0].shape
out1 = cv2.VideoWriter('output_video1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))
height, width, _ = frames2[0].shape
out2 = cv2.VideoWriter('output_video2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "global_cam")
camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
print(f"Camera ID: {camera_id}, Camera Name: {camera_name}")


for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
for frame in frames1:
    out1.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
for frame in frames2:
    out2.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

out.release()
out1.release()
out2.release()

# THIS WORKS AND SAVES THE VIDEO OF THE CAMERA FEED