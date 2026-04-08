#!/usr/bin/env python3
"""
======================================================
 Autonomous Navigation — TurtleBot (ROS / move_base)
======================================================
Map:  map.yaml  +  map.pgm
      resolution : 0.05 m/px
      origin     : (-15.4, -26.6, 0.0)
      size       : 640 x 768 px  ->  32 m x 38.4 m
      X range    : -15.4  to  +16.6
      Y range    : -26.6  to  +11.8

Tour: L1  ->  L2  ->  L3  ->  L1   (fully autonomous, no human input)

Verified locations (all inside map, all in free space):
  L1 (-10.0,  +8.0)  <->  L2 (+10.0,  -5.0) : 23.9 m  OK
  L1 (-10.0,  +8.0)  <->  L3 ( -5.0, -20.0) : 28.4 m  OK
  L2 (+10.0,  -5.0)  <->  L3 ( -5.0, -20.0) : 21.2 m  OK

------------------------------------------------------
PRE-REQUISITES - run each in a SEPARATE terminal first
------------------------------------------------------
  Terminal 1 - Robot base driver:
    roslaunch turtlebot_bringup minimal.launch

  Terminal 2 - AMCL navigation stack:
    roslaunch turtlebot_navigation amcl_demo.launch \
        map_file:=/home/turtlebot/Desktop/map.yaml

  Terminal 3 - RViz (optional, for visualisation):
    roslaunch turtlebot_rviz_launchers view_navigation.launch

  Terminal 4 - This script:
    chmod +x autonomous_navigation.py
    python3 autonomous_navigation.py

IMPORTANT: Before running this script, open RViz and use the
  "2D Pose Estimate" button to click the robot's real starting
  position on the map so AMCL localises correctly.
------------------------------------------------------
"""

import math
import sys

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler
from actionlib_msgs.msg import GoalStatus


# ==================================================================
#  LOCATION DEFINITIONS
#  All coordinates are in the /map frame (metres).
#  Verified to be inside map bounds and in free space.
#
#  x, y : world position
#  yaw  : desired heading on arrival (radians)
#          0.0        -> facing +X  (East)
#          math.pi/2  -> facing +Y  (North)
#          math.pi    -> facing -X  (West)
#         -math.pi/2  -> facing -Y  (South)
# ==================================================================

LOCATIONS = {
    "L1": {"x": -10.0, "y":   8.0, "yaw":  0.0},           # top-left area
    "L2": {"x":  10.0, "y":  -5.0, "yaw": -math.pi / 2},   # right-centre area
    "L3": {"x":  -5.0, "y": -20.0, "yaw":  math.pi},       # bottom area
}

# -- Navigation tour (edit order freely) --
TOUR = ["L1", "L2", "L3", "L1"]

# -- Tuning --
GOAL_TIMEOUT   = 180.0   # seconds to wait per waypoint before giving up
RETRY_ATTEMPTS = 2       # retries on failure before moving to next waypoint
SETTLE_PAUSE   = 2.0     # seconds to pause after arriving at each waypoint


# ==================================================================
#  HELPER FUNCTIONS
# ==================================================================

def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert a yaw angle (radians) to a ROS Quaternion."""
    q = quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def build_goal(location_name: str) -> MoveBaseGoal:
    """Construct a MoveBaseGoal message for the named location."""
    loc  = LOCATIONS[location_name]
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp    = rospy.Time.now()
    goal.target_pose.pose.position.x  = loc["x"]
    goal.target_pose.pose.position.y  = loc["y"]
    goal.target_pose.pose.position.z  = 0.0
    goal.target_pose.pose.orientation = yaw_to_quaternion(loc["yaw"])
    return goal


def status_string(status: int) -> str:
    """Return a human-readable string for a GoalStatus code."""
    return {
        GoalStatus.PENDING:   "PENDING",
        GoalStatus.ACTIVE:    "ACTIVE",
        GoalStatus.PREEMPTED: "PREEMPTED",
        GoalStatus.SUCCEEDED: "SUCCEEDED",
        GoalStatus.ABORTED:   "ABORTED",
        GoalStatus.REJECTED:  "REJECTED",
        GoalStatus.LOST:      "LOST",
    }.get(status, f"UNKNOWN({status})")


# ==================================================================
#  CORE NAVIGATION FUNCTION
# ==================================================================

def navigate_to(client: actionlib.SimpleActionClient,
                location_name: str,
                attempt: int = 1) -> bool:
    """
    Send the robot to `location_name` and block until arrival or timeout.
    Returns True on success, False on failure / timeout.
    """
    loc = LOCATIONS[location_name]
    rospy.loginfo(
        "---  [Attempt %d] Navigating to %s  x=%.2f  y=%.2f  ---",
        attempt, location_name, loc["x"], loc["y"]
    )

    client.send_goal(build_goal(location_name))
    finished = client.wait_for_result(rospy.Duration(GOAL_TIMEOUT))

    if not finished:
        rospy.logwarn("  TIMEOUT after %.0f s -- cancelling goal.", GOAL_TIMEOUT)
        client.cancel_goal()
        return False

    state = client.get_state()
    rospy.loginfo("  move_base result: %s", status_string(state))

    if state == GoalStatus.SUCCEEDED:
        rospy.loginfo("  SUCCESS: Arrived at %s!", location_name)
        return True

    rospy.logwarn("  FAILED: Goal ended with status %s", status_string(state))
    return False


# ==================================================================
#  MAIN
# ==================================================================

def main():
    rospy.init_node("autonomous_navigation", anonymous=False)

    rospy.loginfo("=" * 50)
    rospy.loginfo("  Autonomous Navigation Node -- started")
    rospy.loginfo("  Tour: %s", " -> ".join(TOUR))
    rospy.loginfo("=" * 50)

    # -- Connect to move_base --
    rospy.loginfo("Waiting for /move_base action server ...")
    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)

    if not client.wait_for_server(rospy.Duration(30.0)):
        rospy.logerr(
            "Could not connect to /move_base after 30 s. "
            "Is the navigation stack running?\n"
            "  roslaunch turtlebot_navigation amcl_demo.launch "
            "map_file:=/home/turtlebot/Desktop/map.yaml"
        )
        sys.exit(1)

    rospy.loginfo("Connected to /move_base OK\n")

    # -- Run tour --
    successes = 0
    failures  = 0

    for idx, location_name in enumerate(TOUR):
        if rospy.is_shutdown():
            break

        rospy.loginfo("\n[%d / %d]  Next waypoint -> %s",
                      idx + 1, len(TOUR), location_name)

        reached = False
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            if rospy.is_shutdown():
                break
            reached = navigate_to(client, location_name, attempt)
            if reached:
                break
            if attempt < RETRY_ATTEMPTS:
                rospy.logwarn("  Retrying in 3 s ...")
                rospy.sleep(3.0)

        if reached:
            successes += 1
            rospy.loginfo("  Pausing %.1f s before next waypoint...\n", SETTLE_PAUSE)
            rospy.sleep(SETTLE_PAUSE)
        else:
            failures += 1
            rospy.logerr(
                "  FAILED to reach %s after %d attempt(s). Skipping.\n",
                location_name, RETRY_ATTEMPTS
            )

    # -- Summary --
    rospy.loginfo("=" * 50)
    rospy.loginfo("  Tour complete!")
    rospy.loginfo("  Reached : %d / %d waypoints", successes, len(TOUR))
    rospy.loginfo("  Failed  : %d / %d waypoints", failures,  len(TOUR))
    if failures == 0:
        rospy.loginfo("  ALL waypoints reached successfully!")
    else:
        rospy.logwarn(
            "  Some waypoints were missed. Possible causes:\n"
            "    - Obstacles blocking the path\n"
            "    - Poor AMCL localisation (redo 2D Pose Estimate in RViz)\n"
            "    - move_base costmap needs tuning"
        )
    rospy.loginfo("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted by user.")
