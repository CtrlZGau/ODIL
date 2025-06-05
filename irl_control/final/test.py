from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
import cv2
import numpy as np

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor = SIFT(max_num_keypoints=4096).eval().cuda()  # load the extractor
matcher = LightGlue(features='sift').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = load_image('assets/expert_wrist_cam.jpeg').cuda()
image1 = load_image('assets/cut.jpeg').cuda()

# extract local features
feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
feats1 = extractor.extract(image1)

# match the features
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)

print(points0, points1)

# Step 1: Intrinsics (adjust to your actual resolution if needed)
h, w = 480, 640
fx = fy = 1000
cx = w / 2
cy = h / 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# Step 2: Get matched points as numpy arrays
pts0 = points0.cpu().numpy()
pts1 = points1.cpu().numpy()

H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC)

if H is not None:
    print("Homography:\n", H)

    retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    print(translations)



