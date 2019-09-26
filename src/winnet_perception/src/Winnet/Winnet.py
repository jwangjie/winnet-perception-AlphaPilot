#!/usr/bin/env python
import rospy
from winnet_perception.msg import CNN_out
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty

import cv2
import utils

from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K

TEST_PHASE=0

class Winnet(object):
    def __init__(self,
                 weights_path, target_size=(96, 96),
                 imgs_rootpath="../models"):

        self.pub = rospy.Publisher("cnn_predictions", CNN_out, queue_size=5)

        self.imgs_rootpath = imgs_rootpath

        # Set keras utils
        K.set_learning_phase(TEST_PHASE)

        self.model = load_model(weights_path)
        print("Loaded model from {}".format(weights_path))

        self.target_size = target_size



    def run(self):
        while not rospy.is_shutdown():
            msg = CNN_out()
            msg.header.stamp = rospy.Time.now()
            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message("camera", Image, timeout=10)
                except:
                    pass

            cv_image = utils.callback_img(data, self.target_size, self.imgs_rootpath)

            outs = self.model.predict_on_batch(cv_image[None])

            outs = outs * 48 + 48 # undo the normalization
 
            x1, y1, x2, y2, x3, y3, x4, y4 = outs[0][0], outs[0][1], outs[0][2], outs[0][3], outs[0][4], outs[0][5], outs[0][6], outs[0][7]

            # four corners

            msg.left_up_x = x1
            msg.left_up_y = y1
            msg.right_up_x = x2
            msg.right_up_y = y2
            msg.right_down_x = x3
            msg.right_down_y = y3
            msg.left_down_x = x4
            msg.left_down_y = y4

            # calculate the waypoints ahead and after the window


            self.pub.publish(msg)  # publish eight numbers 