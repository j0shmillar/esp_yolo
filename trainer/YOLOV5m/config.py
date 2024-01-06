import os
from pathlib import Path
import albumentations as A
import torch.cuda

parent_dir = Path(__file__).parent
ROOT_DIR = os.path.join(parent_dir, "datasets", "ROAD")

# if no yaml file, this must be manually inserted
# nc is number of classes (int)
nc = None
# list containing the labels of classes: i.e. ["cat", "dog"]
labels = None

FIRST_OUT = 48

CLS_PW = 1.0
OBJ_PW = 1.0

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640

CONF_THRESHOLD = 0.5  # to get all possible bboxes, trade-off metrics/speed --> we choose metrics
NMS_IOU_THRESH = 0.6
# for map 50
MAP_IOU_THRESH = 0.5

# triple check what anchors REALLY are

# ANCHORS = [
#     [(10, 13), (16, 30), (33, 23)],  # P3/8
#     [(30, 61), (62, 45), (59, 119)],  # P4/16
#     [(116, 90), (156, 198), (373, 326)]  # P5/32#
# ]
ANCHORS = [ [(36,36), (78,78), (139,135)]]


TRAIN_TRANSFORMS = A.Compose(
    [
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.4),
        A.Transpose(p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-20, 20), p=0.7),
        A.Blur(p=0.05),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ChannelShuffle(p=0.05),
    ],
    bbox_params=A.BboxParams("yolo", min_visibility=0.4, label_fields=[],),
)

FLIR = [
    'car',
    'person'
]

ROAD = ["trafficlight","stop", "speedlimit","crosswalk"]

nc = len(ROAD)
labels = ROAD
