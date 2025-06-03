import time as time_lib
from typing import Dict

import numpy as np
from gymnasium.spaces import Box

from irl_control.mujoco_gym_app import MujocoGymAppHighFidelity
from irl_control.utils.target import Target
import cv2
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict, annotate

import cv2

def annotate(image_source, boxes, logits, phrases):
    annotated_image = image_source.copy()

    for box, logit, phrase in zip(boxes, logits, phrases):
        x1, y1, x2, y2 = map(int, box)

        label = f"{phrase} ({logit:.2f})"

        # Draw the rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        # Put the label text
        font_scale = 0.5
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # Make a filled rectangle behind the text for better readability
        cv2.rectangle(annotated_image, (x1, y1 - text_height - 4), (x1 + text_width, y1), (0, 0, 255), -1)
        cv2.putText(annotated_image, label, (x1, y1 - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated_image


model = load_model("/Users/gautham/desktop/coding/ml/mlscratch/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "./groundingdino_swint_ogc.pth")
IMAGE_PATH = "assets/demo.jpeg"
TEXT_PROMPT = "red cube"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
device = "cpu"



image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device="cpu"
)

boxes = boxes.cpu().numpy() if hasattr(boxes, 'cpu') else boxes

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame) 


