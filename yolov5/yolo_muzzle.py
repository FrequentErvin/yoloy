from pathlib import Path

import torch
from models.common import DetectMultiBackend
from utils.general import (increment_path, non_max_suppression)
from utils.plots import save_one_box
from PIL import Image
import numpy as np
class YoloMuzzle:
    device = torch.device('cpu')

    model = DetectMultiBackend('best.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
    save_dir = increment_path(Path("") / "name", exist_ok=True)
    visualize = increment_path(save_dir / Path("path").stem, mkdir=True)

    def detect(self, img, img0s):

        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        print('ker')
        print(im)
        pred = self.model(im, augment=False, visualize=False)
        print('dlon')
        print(pred)
        pred = non_max_suppression(pred)
        print('tlon')
        print(pred)
        for i, det in enumerate(pred):
            for *xyxy, conf, cls in reversed(det):
                print("Zaba")

        print("maca")
        print("prije save one box", xyxy)
        print("IMAGE BEFORE SAVE ONE BOX", img)
        img0 = img0s.copy()
        imc = img0.copy()
        cropped = save_one_box(xyxy, imc, BGR=False, save=False)
        # savedCrop = Image.fromarray(cropped, 'RGB')
        # savedCrop.save('nesto.png')
        # savedCrop.show()
        # print("cropped", cropped)
        return cropped