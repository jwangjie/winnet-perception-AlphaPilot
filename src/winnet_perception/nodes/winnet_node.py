#!/usr/bin/env python

import rospy
from Winnet import Winnet
import os, datetime

def run_network():

    rospy.init_node('winnet', anonymous=True)

    # LOAD ROS PARAMETERS 

    weights_model_path = rospy.get_param("~weights_path")
    
    imgs_rootpath = None
    target_size = rospy.get_param("~target_size", '96, 96').split(',')
    target_size = tuple([int(t) for t in target_size])


    # BUILD NETWORK CLASS

    network = Winnet.Winnet(weights_model_path, target_size, imgs_rootpath)

    # RUN NETWORK
    network.run()

if __name__ == "__main__":
    run_network()

