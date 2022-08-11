from ResNet import ResNet
from ResNet import Bottleneck

from collections import namedtuple

import torch

from torchvision.transforms import ToTensor

from torchvision import transforms

from flask import Flask, request

from flask_restful import Resource, Api
from flask_cors import CORS
import flask.scaffold

import numpy as np
from io import BytesIO
from PIL import Image

from yolo_muzzle import YoloMuzzle
import numpy as np
from utils.augmentations import letterbox
import cv2

#flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
PATH = "yolov5/modelRES5"

resnet50_config = ResNetConfig(block=Bottleneck,
                               n_blocks=[3, 4, 6, 3],
                               channels=[64, 128, 256, 512])

model = ResNet(resnet50_config, 20)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

model.eval()

app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

krave = ['0706',
'0728',
'0844',
'0858',
'0859',
'0892',
'0894',
'0899',
'0965',
'0986',
'1271',
'3364',
'3376',
'3588',
'3709',
'3717',
'3794',
'3821',
'3831',
'3876'
]
# prediction api call
class prediction(Resource):

    objectDetectionModel = YoloMuzzle()
    def post(self):
        f = request.files['image']
        img = Image.open(BytesIO(f.read())).convert('RGB')
        img.show()
        img0s = np.array(img)  # expecting cv2 imread
        return self.processNumpyArray(img0s)


    def processNumpyArray(self, img0s):
        # Image.fromarray(img0s, 'RGB').show()
        # YOLO PART

        img = letterbox(img0s, 640, 32, auto=True)[0]
        Image.fromarray(img, 'RGB').show()
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        pred = self.objectDetectionModel.detect(img, img0s)  # returns numpy array
        # YOLO END

        img_tensor = preprocess(pred).float()  # expecting PIL image for input here
        img_tensor = img_tensor.unsqueeze_(0)

        predicted = model(img_tensor)
        print(predicted)
        print(predicted[1])
        return krave[predicted[0][0].argmax().item()]



api.add_resource(prediction, '/api/prediction/')

if __name__ == '__main__':
    app.run(debug=True)
