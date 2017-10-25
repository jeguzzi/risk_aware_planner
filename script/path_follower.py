#!/usr/bin/env python


import rospy

import numpy as np
from shapely.geometry import LineString, Point
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import (PoseStamped, Twist, Vector3)
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion


def array_from_msg(point_msg):
    return np.array(point_from_msg(point_msg))


def array3_from_msg(point_msg):
    return np.array([point_msg.x, point_msg.y, point_msg.z])


def point_from_msg(point_msg):
    return [point_msg.x, point_msg.y, point_msg.z]


def quaternion_from_msg(quaternion_msg):
    return [quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w]


def yaw_from_msg(quaternion_msg):
    return euler_from_quaternion(quaternion_from_msg(quaternion_msg))[2]


def angle_difference(angle_1, angle_2):
    a1, a2 = np.unwrap(np.array([angle_1, angle_2]))
    return a2 - a1


def get_transform(tf_buffer, from_frame, to_frame):
    try:
        return tf_buffer.lookup_transform(
            from_frame, to_frame, rospy.Time(0), rospy.Duration(0.1)
        )
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as e:
        rospy.logerr(e)
        return None


def path_in_frame(tf_buffer, path, frame_id):
    t = get_transform(tf_buffer, frame_id, path.header.frame_id)
    if not t:
        return None
    msg = Path(header=path.header)
    msg.header.frame_id = frame_id
    msg.poses = [tf2_geometry_msgs.do_transform_pose(pose, t) for pose in path.poses]
    return msg


def pose_in_frame(tf_buffer, pose_s, frame_id):
    # rospy.loginfo('pose_in_frame %s %s %s', tf_buffer, pose_s, frame_id)
    t = get_transform(tf_buffer, frame_id, pose_s.header.frame_id)
    if not t:
        return None
    return tf2_geometry_msgs.do_transform_pose(pose_s, t)


def normalize_s(s, max_s, loop):
    if s < 0:
        if loop:
            s = s + max_s
        else:
            s = 0
    elif s > max_s:
        if loop:
            s = s - max_s
        else:
            s = max_s
    return s


def t_speed(x0, x1, tau, max_speed):
    v = (x1 - x0) / tau
    s = np.linalg.norm(v)
    if s > max_speed:
        v = v / s * max_speed
    return v


def t_angular_speed(x0, x1, tau, max_speed):
    v = angle_difference(x0, x1) / tau
    s = np.abs(v)
    if s > max_speed:
        v = v / s * max_speed
    return v


class PathFollower(object):
    """docstring for PathFollower."""
    def __init__(self):

        rospy.init_node("path_follower")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.curve = None
        self.path = None
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.delta = rospy.get_param("~delta", 0.5)
        self.min_distance = rospy.get_param("~min_distance", 0.5)
        self.tau = rospy.get_param("~tau", 0.5)
        self.target_speed = rospy.get_param("~max_speed", 0.3)
        self.target_angular_speed = rospy.get_param("~max_angular_speed", 0.3)
        self.k = rospy.get_param("~k", 1.0)
        rate = rospy.get_param('~rate', 5.0)
        self.min_dt = 1.0 / rate
        self.last_t = rospy.Time.now()
        self.pub_twist = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("selected_path", Path, self.has_updated_path)
        rospy.Subscriber("pose", PoseStamped, self.has_updated_pose)
        rospy.Subscriber("target", PoseStamped, self.has_updated_target)
        rospy.spin()

    def has_updated_target(self, msg):
        self.stop()

    def should_send(self):
        dt = rospy.Time.now() - self.last_t
        return dt.to_sec() > self.min_dt

    def stop(self, msg=None):
        self.path = None
        self.curve = None
        self.pub_twist.publish(Twist())

    @property
    def target_point(self):
        if self.path and self.path.poses:
            return self.path.poses[-1].pose.position
        return None

    def has_arrived(self, point):
        if not self.path:
            return True
        distance = np.linalg.norm(array_from_msg(self.target_point)[:2] -
                                  array_from_msg(point)[:2])
        return distance < self.min_distance

    def has_updated_path(self, msg):
        path = path_in_frame(self.tf_buffer, msg, self.frame_id)
        if not msg.poses or not path:
            rospy.loginfo("Got invalid/empty path, will stop")
            self.stop()
            return
        rospy.loginfo("Got new path")
        self.curve = LineString([point_from_msg(pose.pose.position) for pose in path.poses])
        self.ps = np.array(self.curve)
        self.ls = np.linalg.norm(np.diff(self.ps, axis=0), axis=1)
        self.cs = np.cumsum(self.ls)
        self.path = path

    def target_along_path(self, current_point):
        cp = Point(current_point)
        s = self.curve.project(cp)
        s = s + self.delta
        if s > self.cs[-1]:
            return self.ps[-1]
        return np.array(self.curve.interpolate(s))

    def target_twist_along_path(self, pose_s):
        if not self.path:
            return None
        current_point = array_from_msg(pose_s.pose.position)
        current_yaw = yaw_from_msg(pose_s.pose.orientation)
        target_point = self.target_along_path(current_point)
        delta = target_point - current_point
        target_yaw = np.arctan2(delta[1], delta[0])
        target_angular_speed = t_angular_speed(current_yaw, target_yaw, self.tau,
                                               self.target_angular_speed)
        f = max(0, (1 - self.k * abs(target_angular_speed) / self.target_angular_speed))
        target_speed = self.target_speed * f
        return Twist(linear=Vector3(target_speed, 0, 0),
                     angular=Vector3(0, 0, target_angular_speed))

    def has_updated_pose(self, msg):
        if self.path:
            pose_s = pose_in_frame(self.tf_buffer, msg, self.frame_id)
            if not pose_s:
                rospy.logerr('Could not transform pose %s to frame %s', msg, self.frame_id)
                return
            point = pose_s.pose.position
            if self.has_arrived(point):
                rospy.loginfo('Has arrived, will stop')
                self.stop()
                return
            target_twist = self.target_twist_along_path(pose_s)
            if not target_twist:
                rospy.logerr('No target twist')
                return
            if self.should_send():
                self.last_t = rospy.Time.now()
                if target_twist:
                    self.pub_twist.publish(target_twist)


if __name__ == '__main__':
    PathFollower()
