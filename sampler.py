import cv2
import numpy as np

def getFrames(fname):
    vidcap = cv2.VideoCapture(fname)
    success,image = vidcap.read()
    success = True
    frames = []
    while success:
      success,image = vidcap.read()
      if success:
          rescaled = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
          frames.append(cv2.cvtColor(rescaled, cv2.COLOR_BGR2GRAY))

    return np.array(frames)
