#!/usr/bin/env python3
"""
Green Ball Follower for TurtleBot2
- RGB camera  : /usb_cam/image_raw                    (usb_cam driver)
- Depth camera: /camera/depth_registered/image_raw    (Astra, registered to RGB frame)
- Drives the robot to follow a green ball at ~1 metre distance

FIXES APPLIED
─────────────
  Bug 1 │ Renamed detect_red_ball → detect_green_ball (was misleading).
  Bug 2 │ Subscribe to /camera/depth_registered/image_raw (projected into
         │ the RGB camera frame via depth_image_proc/register) instead of
         │ the raw depth topic, so RGB pixel coords are valid in the depth
         │ image without any manual scaling.
  Bug 3 │ Replaced two independent Subscribers with
         │ message_filters.ApproximateTimeSynchronizer so the RGB and depth
         │ frames are guaranteed to be temporally matched before drive() runs.
  Bug 4 │ Moved rospy.loginfo() to AFTER cmd.linear.x is computed so the
         │ log always shows the real velocity that is published.

LAUNCH-FILE ADDITION REQUIRED (Bug 2)
──────────────────────────────────────
Add this node to your launch file so the depth image is reprojected into
the RGB camera frame before this script subscribes to it:

  <node pkg="depth_image_proc" type="register" name="depth_register">
    <remap from="rgb/camera_info"
           to="/usb_cam/camera_info"/>
    <remap from="depth/camera_info"
           to="/camera/depth/camera_info"/>
    <remap from="depth/image_rect"
           to="/camera/depth/image_raw"/>
    <remap from="depth_registered/image_rect"
           to="/camera/depth_registered/image_raw"/>
  </node>
"""

import rospy
import message_filters          # FIX 3 — time-synchronised callbacks
import numpy as np
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


class GreenBallFollower:
    def __init__(self):
        rospy.init_node('green_ball_follower', anonymous=True)

        # ── Parameters (tunable at runtime via rosparam) ──────────────────
        self.desired_dist  = rospy.get_param('~desired_dist',  1.0)   # metres
        self.tol_dist      = rospy.get_param('~tol_dist',      0.05)  # metres deadband
        self.tol_ang       = rospy.get_param('~tol_ang',       0.03)  # rad   deadband

        self.p_gain_lin    = rospy.get_param('~p_gain_lin',    0.4)
        self.p_gain_ang    = rospy.get_param('~p_gain_ang',    0.8)

        self.max_speed     = rospy.get_param('~max_speed',     0.2)   # m/s
        self.max_rot       = rospy.get_param('~max_rot',       0.8)   # rad/s

        self.min_ball_area = rospy.get_param('~min_ball_area', 300)   # pixels²

        rgb_topic   = rospy.get_param('~rgb_topic',   '/usb_cam/image_raw')
        # FIX 2 — use the depth image registered (reprojected) into the RGB
        # camera frame so that RGB pixel coords are valid inside it.
        depth_topic = rospy.get_param('~depth_topic',
                                      '/camera/depth_registered/image_raw')
        vel_topic   = rospy.get_param('~vel_topic',   '/cmd_vel_mux/input/navi')

        # ── Internal state ─────────────────────────────────────────────────
        self.bridge    = CvBridge()
        self.img_width = None

        # ── Publisher ──────────────────────────────────────────────────────
        self.vel_pub = rospy.Publisher(vel_topic, Twist, queue_size=1)

        # ── FIX 3: Synchronised subscribers ───────────────────────────────
        # ApproximateTimeSynchronizer pairs RGB and depth frames whose
        # timestamps are within `slop` seconds of each other.  Without this,
        # the depth stored in a plain Subscriber callback could be hundreds of
        # milliseconds older than the current RGB frame, causing the robot to
        # react to a depth reading that no longer matches the ball position.
        rgb_sub   = message_filters.Subscriber(rgb_topic,   Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=5,
            slop=0.05          # accept pairs within 50 ms of each other
        )
        self.sync.registerCallback(self.sync_callback)

        rospy.loginfo("Green Ball Follower ready.")
        rospy.loginfo("  RGB  topic : %s", rgb_topic)
        rospy.loginfo("  Depth topic: %s", depth_topic)
        rospy.spin()

    # ── Synchronised RGB + depth callback (FIX 3) ──────────────────────────
    def sync_callback(self, rgb_msg, depth_msg):
        """Called only when a matched RGB+depth pair is available."""

        # ── Decode depth frame ─────────────────────────────────────────────
        try:
            if depth_msg.encoding == '16UC1':
                raw = self.bridge.imgmsg_to_cv2(depth_msg,
                                                desired_encoding='16UC1')
                depth_image = raw.astype(np.float32) / 1000.0  # mm → m
            else:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg,
                                                        desired_encoding='32FC1')
        except CvBridgeError as e:
            rospy.logwarn("Depth bridge error: %s", e)
            return

        # ── Decode RGB frame ───────────────────────────────────────────────
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg,
                                              desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logwarn("RGB bridge error: %s", e)
            return

        self.img_width = frame.shape[1]
        img_height     = frame.shape[0]

        # ── Detect green ball (FIX 1: method renamed) ──────────────────────
        cx, cy, mask = self.detect_green_ball(frame)

        # ── Visualise ───────────────────────────────────────────────────────
        debug = frame.copy()
        if cx is not None:
            cv2.circle(debug, (cx, cy), 10, (0, 255, 0), -1)
            cv2.line(debug,
                     (self.img_width // 2, 0),
                     (self.img_width // 2, img_height),
                     (255, 0, 0), 1)
        cv2.imshow("Green Ball Tracker", debug)
        cv2.imshow("Green Mask",         mask)
        cv2.waitKey(1)

        # ── Drive ───────────────────────────────────────────────────────────
        self.drive(cx, cy, depth_image)

    # ── Ball detection using HSV thresholding (FIX 1: renamed) ──────────────
    def detect_green_ball(self, frame):
        """
        Returns (cx, cy, mask).
        Green sits in the middle of the HSV hue wheel (~35–85°).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35,  50,  50])
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
    def get_depth_at(self, depth_image, cx, cy):
        """
        Sample a small patch around (cx, cy) in the registered depth image.
        Because Bug 2 is fixed, (cx, cy) from the RGB frame are valid
        coordinates inside this depth image.
        """
        h, w = depth_image.shape[:2]
        x0, x1 = max(cx - 5, 0), min(cx + 5, w)
        y0, y1 = max(cy - 5, 0), min(cy + 5, h)

        patch = depth_image[y0:y1, x0:x1]
        valid = patch[(patch > 0.1) & (patch < 10.0)]

        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ── P-controller ─────────────────────────────────────────────────────────
    def drive(self, cx, cy, depth_image):
        cmd = Twist()   # default: all zeros → stop

        if cx is None or self.img_width is None:
            rospy.loginfo_throttle(2, "No green ball detected — stopping.")
            self.vel_pub.publish(cmd)
            return

        # ── Angular control: centre the ball horizontally ───────────────────
        # Normalised error: [-0.5, +0.5]
        ang_error = (self.img_width / 2.0 - cx) / self.img_width

        if abs(ang_error) > self.tol_ang:
            cmd.angular.z = ang_error * self.p_gain_ang
            cmd.angular.z = max(-self.max_rot,
                                min(self.max_rot, cmd.angular.z))

        # ── Linear control: maintain desired distance ────────────────────────
        depth = self.get_depth_at(depth_image, cx, cy)

        if depth is None:
            rospy.logwarn_throttle(2,
                "No valid depth reading at ball centre — rotating only.")
        else:
            dist_error = depth - self.desired_dist

            if abs(dist_error) > self.tol_dist:
                cmd.linear.x = dist_error * self.p_gain_lin
                cmd.linear.x = max(-self.max_speed,
                                   min(self.max_speed, cmd.linear.x))

            # FIX 4 — log AFTER cmd.linear.x is computed so it shows the real
            # value that will be published, not the 0.0 default.
            rospy.loginfo(
                "depth: %.2f m | dist_err: %.2f | ang_err: %.3f | "
                "lin: %.2f | ang: %.2f",
                depth, dist_error, ang_error,
                cmd.linear.x, cmd.angular.z
            )

        self.vel_pub.publish(cmd)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        GreenBallFollower()
    except rospy.ROSInterruptException:
        pass
