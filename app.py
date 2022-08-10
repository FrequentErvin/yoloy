from ResNet import ResNet
from ResNet import Bottleneck

from collections import namedtuple

import torch

from torchvision.transforms import ToTensor

from torchvision import transforms

from flask import Flask, request
import flask.scaffold

from flask_restful import Resource, Api
from flask_cors import CORS
import flask.scaffold

import numpy as np
from io import BytesIO
from PIL import Image

#flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
PATH = "modelRES5"

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


# prediction api call
class prediction(Resource):
    def post(self):
        f = request.files['image']
        img = Image.open(BytesIO(f.read())).convert('RGB')

        img_tensor = preprocess(img).float()
        img_tensor = img_tensor.unsqueeze_(0)

        predicted = model(img_tensor)
        return predicted[0][0].argmax().item()



api.add_resource(prediction, '/api/prediction/')

if __name__ == '__main__':
    app.run(debug=True)
