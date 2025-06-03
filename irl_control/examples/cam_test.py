import mujoco
import numpy as np

# Load model and data
model = mujoco.MjModel.from_xml_path('assets/kinect_environment.xml')
data = mujoco.MjData(model)

# List of camera names as defined in your XML
camera_names = ['wrist_cam_right', 'wrist_cam_left', 'global_cam']

# Create renderer
renderer = mujoco.Renderer(model)

images = {}
for cam_name in camera_names:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render()
    images[cam_name] = img  # img is a numpy array (H, W, 3)
# Display or save the images
for cam_name, img in images.items():
    print(f"Image from {cam_name} has shape: {img.shape}")
# You can use matplotlib to display the images or save them using OpenCV or PIL
import matplotlib.pyplot as plt
for cam_name, img in images.items():
    plt.imshow(img)
    plt.title(cam_name)
    plt.axis('off')
    plt.show()
    # Optionally save or display the image here
