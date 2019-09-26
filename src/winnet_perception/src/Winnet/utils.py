from cv_bridge import CvBridge, CvBridgeError
from tensorflow.keras.models import model_from_json

import cv2
import numpy as np
import rospy

bridge = CvBridge()

def callback_img(data, target_size, rootpath):
    try:
        image_type = data.encoding
        
        #print(image_type)
        img = bridge.imgmsg_to_cv2(data, image_type)
    except CvBridgeError, e:
        print e

    img = cv2.resize(img, target_size)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)                          # read as np array
    img = img / 255.0                              # normalize the image
    #img = img.reshape(-1, target_size[0], target_size[1], 3)   # Add another dimension for tensorflow

    if rootpath:
        temp = rospy.Time.now()
        cv2.imwrite("{}/{}.jpg".format(rootpath, temp), img)

    return np.asarray(img, dtype=np.float32) * np.float32(1.0/255.0)


def jsonToModel(json_model_path):
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    return model
