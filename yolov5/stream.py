import cv2
from app import prediction
from PIL import Image
import numpy as np


instance = prediction()

try:
    stream = cv2.VideoCapture(RTSP_URL)
    ret, frame = stream.read()
    # print("frame", frame)
    # cv2.imshow("frame", frame)
    cropped = frame[200:840, 800:1280]
    cv2.imwrite('nesto.jpg', cropped)
    # cv2.imshow("Frame", frame)
    pilImage = Image.open('nesto.jpg').convert('RGB')
    # pilImage.show()
    numpyImage = np.array(pilImage)

    result = instance.processNumpyArray(numpyImage)
    print(result)

finally:
    stream.release()
    cv2.destroyAllWindows()


