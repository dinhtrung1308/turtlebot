#!/usr/bin/env python3
"""
Red Ball Follower for TurtleBot2
- RGB camera  : /usb_cam/image_raw        (usb_cam driver)
- Depth camera: /camera/depth/image_raw   (Astra)
- Drives the robot to follow a red ball at ~1 metre distance
"""

import rospy
import math
import numpy as np
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class RedBallFollower:
    def __init__(self):
        rospy.init_node('red_ball_follower', anonymous=True)

        # ── Parameters (tunable at runtime via rosparam) ──────────────────
        self.desired_dist   = rospy.get_param('~desired_dist',  1.0)   # metres
        self.tol_dist       = rospy.get_param('~tol_dist',      0.05)  # metres deadband
        self.tol_ang        = rospy.get_param('~tol_ang',       0.03)  # rad   deadband

        self.p_gain_lin     = rospy.get_param('~p_gain_lin',    0.4)
        self.p_gain_ang     = rospy.get_param('~p_gain_ang',    0.8)

        self.max_speed      = rospy.get_param('~max_speed',     0.2)   # m/s
        self.max_rot        = rospy.get_param('~max_rot',       0.8)   # rad/s

        self.min_ball_area  = rospy.get_param('~min_ball_area', 300)   # pixels²

        rgb_topic   = rospy.get_param('~rgb_topic',   '/usb_cam/image_raw')
        depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')
        vel_topic   = rospy.get_param('~vel_topic',   '/cmd_vel_mux/input/navi')

        # ── Internal state ─────────────────────────────────────────────────
        self.bridge      = CvBridge()
        self.ball_cx     = None   # pixel column of ball centre
        self.ball_cy     = None   # pixel row    of ball centre
        self.img_width   = None
        self.depth_image = None   # latest depth frame (float32, metres)

        # ── ROS interfaces ─────────────────────────────────────────────────
        self.vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=1)

        rospy.Subscriber(rgb_topic,   Image, self.rgb_callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)

        rospy.loginfo("Red Ball Follower ready.")
        rospy.spin()

    # ── Depth callback ──────────────────────────────────────────────────────
    def depth_callback(self, msg):
        try:
            # Astra publishes 32FC1 (metres) or 16UC1 (millimetres)
            if msg.encoding == '16UC1':
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                self.depth_image = raw.astype(np.float32) / 1000.0   # mm → m
            else:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except CvBridgeError as e:
            rospy.logwarn("Depth bridge error: %s", e)

    # ── RGB callback ─────────────────────────────────────────────────────────
    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logwarn("RGB bridge error: %s", e)
            return

        self.img_width = frame.shape[1]
        img_height     = frame.shape[0]

        # ── 1. Detect red ball ──────────────────────────────────────────────
        cx, cy, mask = self.detect_red_ball(frame)

        # ── 2. Visualise ────────────────────────────────────────────────────
        debug = frame.copy()
        if cx is not None:
            cv2.circle(debug, (cx, cy), 10, (0, 255, 0), -1)
            cv2.line(debug, (self.img_width // 2, 0),
                            (self.img_width // 2, img_height), (255, 0, 0), 1)
        cv2.imshow("Red Ball Tracker", debug)
        cv2.imshow("Red Mask",         mask)
        cv2.waitKey(1)

        # ── 3. Drive ────────────────────────────────────────────────────────
        self.ball_cx = cx
        self.ball_cy = cy
        self.drive()

    # ── Ball detection using HSV thresholding ────────────────────────────────
    def detect_red_ball(self, frame):
        """
        Returns (cx, cy, mask).
        Red wraps around 0°/360° in HSV, so we use two ranges.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_ball_area:
            return None, None, mask

        M  = cv2.moments(largest)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy, mask

    # ── Get depth at ball centre ─────────────────────────────────────────────
    def get_depth_at(self, cx, cy):
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]
        x0, x1 = max(cx - 5, 0), min(cx + 5, w)
        y0, y1 = max(cy - 5, 0), min(cy + 5, h)

        patch = self.depth_image[y0:y1, x0:x1]
        valid = patch[(patch > 0.1) & (patch < 10.0)]   # filter invalid readings

        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ── P-controller ─────────────────────────────────────────────────────────
    def drive(self):
        cmd = Twist()   # default: all zeros → stop

        if self.ball_cx is None or self.img_width is None:
            rospy.loginfo_throttle(2, "No red ball detected — stopping.")
            self.vel_pub.publish(cmd)
            return

        # ── Angular control: centre the ball horizontally ───────────────────
        # Normalised error: [-0.5, +0.5]
        ang_error = (self.img_width / 2.0 - self.ball_cx) / self.img_width

        if abs(ang_error) > self.tol_ang:
            cmd.angular.z = ang_error * self.p_gain_ang
            cmd.angular.z = max(-self.max_rot,
                                min(self.max_rot, cmd.angular.z))

        # ── Linear control: maintain desired distance ────────────────────────
        depth = self.get_depth_at(self.ball_cx, self.ball_cy)

        if depth is None:
            rospy.logwarn_throttle(2, "No valid depth reading — rotating only.")
        else:
            dist_error = depth - self.desired_dist

            rospy.loginfo("depth: %.2f m | dist_err: %.2f | ang_err: %.3f | "
                          "lin: %.2f | ang: %.2f",
                          depth, dist_error, ang_error,
                          cmd.linear.x, cmd.angular.z)

            if abs(dist_error) > self.tol_dist:
                cmd.linear.x = dist_error * self.p_gain_lin
                cmd.linear.x = max(-self.max_speed,
                                   min(self.max_speed, cmd.linear.x))

        self.vel_pub.publish(cmd)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        RedBallFollower()
    except rospy.ROSInterruptException:
        pass
