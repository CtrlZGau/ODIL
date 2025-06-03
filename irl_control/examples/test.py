import numpy as np
import matplotlib.pyplot as plt
import cv2

# Create a blank image (360 rows x 480 columns, 3 color channels)
image_height, image_width = 360, 480
image = cv2.imread("assets/demo.jpeg")

# Pixel coordinates to plot
u, v = 239, 305

# Mark the pixel on the image with a red dot
image[v, u] = [255, 0, 0]  # Red color (BGR)

# Plot the image
plt.imshow(image)
plt.title(f"Pixel point at (u={u}, v={v})")
plt.axis('on')  # Show axes for reference
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
plt.show()
