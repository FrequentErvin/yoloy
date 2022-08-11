from yolo_muzzle import YoloMuzzle
import numpy as np
from utils.augmentations import letterbox
import cv2

model = YoloMuzzle()
imc = cv2.imread("yolov5/IMG_8197.jpeg")
img = letterbox(imc, 640, 32, auto=True)[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
pred = model.detect(img, imc)
