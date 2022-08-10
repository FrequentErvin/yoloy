from yolo_muzzle import YoloMuzzle
import numpy as np
from utils.augmentations import letterbox
import cv2


model = YoloMuzzle()
img0s = cv2.imread("IMG_8197.jpeg")
img = letterbox(img0s, 640, 32, auto=True)[0]
img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
img = np.ascontiguousarray(img)
pred = model.detect(img, img0s)

#pred.show()
