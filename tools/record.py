#! /usr/bin/python
import roslib
roslib.load_manifest('slam_record')
import sys
import rospy
import cv2
import numpy as np
import pdb
from matplotlib import pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os

class image_converter():
    def __init__(self):
        cv2.namedWindow('Image color')
        cv2.namedWindow('Image depth')
        self.bridge = CvBridge()
        self.image_sub_color = rospy.Subscriber('/camera/rgb/image_color',Image, self.callback_color)
        self.image_sub_depth = rospy.Subscriber('/camera/depth_registered/image_raw', Image, self.callback_depth)
        os.system( "mkdir rgb dep")
        rospy.rostime.set_rostime_initialized( True )
        print 'slam recorder is waiting for an image.'

    def callback_color(self, data):
        try:
            cv_image = np.asarray(self.bridge.imgmsg_to_cv(data, 'passthrough'))
        except CvBridgeError, e:
            print e
        time = rospy.rostime.get_time()
        cv2.imshow('Image color', cv_image)
        cv2.waitKey(1)
        cv2.imwrite('./rgb/'+str(time)+'.png', cv_image)
            
    def callback_depth(self, data):
        try:
            cv_image = np.asarray(self.bridge.imgmsg_to_cv(data, 'passthrough'))
        except CvBridgeError, e:
            print e
        time = rospy.rostime.get_time()
        cv2.imwrite('./dep/'+str(time)+'.png', cv_image)
    
if __name__=='__main__':
    ic = image_converter()
    rospy.init_node('image_reader', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "shutting down"
    cv2.destroyAllWindows()
