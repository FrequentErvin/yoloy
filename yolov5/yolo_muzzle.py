from pathlib import Path

import torch
from models.common import DetectMultiBackend
from utils.general import (increment_path, non_max_suppression)
from utils.plots import save_one_box
from PIL import Image

class YoloMuzzle:
    device = torch.device('cpu')

    model = DetectMultiBackend('best.pt', device=device, dnn=False, data='data/coco128.yaml', fp16=False)
    # save_dir = increment_path(Path("") / "name", exist_ok=True)
    # visualize = increment_path(save_dir / Path("path").stem, mkdir=True)

    def detect(self, img, img0s):
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        prediction = self.model(im, augment=False, visualize=False)
        prediction = non_max_suppression(prediction)
        for *xyxy, conf, cls in reversed(prediction[0]):
            print("")
        img0 = img0s.copy()
        imc = img0.copy()
        cropped = save_one_box(xyxy, imc, BGR=False, save=False)
        # savedCrop = Image.fromarray(cropped, 'RGB')
        # savedCrop.show()
        return cropped