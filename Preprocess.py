import cv2
import numpy as np
def preprocess(image):
        #image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #height, width = image.shape[0:2]
        
        image_cropped = image[60:130, :]
        image_resized = cv2.resize(image_cropped, (160, 70))
        return image_resized
     
def flip(image):
    return image
    #return cv2.flip(image, 1)

