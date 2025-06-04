import cv2
import pysift
import numpy as np

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

image = cv2.imread("assets/expert_wrist.png")
blue_mask = segment(image)
cv2.imshow("Segmented Blue", blue_mask)