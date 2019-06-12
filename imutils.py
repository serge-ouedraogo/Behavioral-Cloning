import cv2
import numpy as np
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image_cropped = image[50:20, :]
    return image
     
def flip(image):
    return cv2.flip(image, 1)

