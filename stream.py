import rtsp

import cv2

USERNAME = "admin"
PASSWORD = "A1s2d3f4"

RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@192.168.0.34:554/ISAPI/Streaming/Channels/101"

#client = rtsp.Client(rtsp_server_uri=RTSP_URL, verbose=True)

#client.preview()

#while not client.isOpened():
    #print("Waiting...")

stream = cv2.VideoCapture(RTSP_URL)

while True:
    ret, frame = stream.read()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        stream.release()
        cv2.destroyAllWindows()
