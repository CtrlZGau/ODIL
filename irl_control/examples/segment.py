import cv2
import numpy as np

# Load image
image = cv2.imread("assets/demo.jpeg")

# Convert to HSV color space
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

# Save the output
cv2.imwrite("segmented_blue_shades.png", segmented_blue)
cv2.imwrite("binary_blue_mask.png", blue_mask)
print("Saved blue-shade segmentation as 'segmented_blue_shades.png'")
