from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import numpy as np

from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros

from geometry_msgs.msg import PoseArray, Pose, LaserScan

import rclpy

"""
Things that need to be done:

1. Motion model and snesor model are done and they pass all the unit tests
2. We need to publish all the things to the right places (expect significant debugging)
3. We implemented the mel_average function but we are not sure if it is correct
4. We implemented the pose_callback function but we are not sure if it is correct
5. We added inimai's publish_average_pose function but we are not sure if it is correct

"""

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")
        self.dT = 1/20.0

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
       
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('base_link_topic', "/base_link_pf")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        base_link_topic = self.get_parameter("base_link_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        self.base_link_sub = self.create_subscription(Odometry, base_link_topic,
                                                 self.base_link_callback,
                                                 1)
        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.debug_particles = self.create_publisher(PoseArray, "/pf/particles", 1)
        self.base_link_pub = self.create_publisher()
        self.timer = self.create_timer(self.dT, self.timer_callback)
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

        # Initialize the models
        self.motion_model = MotionModel(self,std_dev_=0.05)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
        self.initiated = False
        self.particles = None
   

    def timer_callback(self):
        if self.initiated:
            #particles generated
            new_particles = self.motion_model.evaluate(self.particles, self.odometry)
            probs = self.sensor_model(new_particles, self.scans)
            # Resample particles based on the probabilities
            self.particles = np.random.choice(new_particles, size=np.shape(new_particles)[0],p=probs)

            particle_msg = PoseArray()
            particle_msg.header.frame_id = self.particle_filter_frame
            particle_msg.header.stamp = self.get_clock().now().to_msg()

            poses = []
            for p in self.particles:
                pose = Pose()
                pose.position.x = p[0]
                pose.position.y = p[1]
                pose.orientation = quaternion_from_euler(0, 0, p[2])
                poses.append(pose)
            particle_msg.poses = poses
            self.debug_particles.publish(particle_msg)     

            mle_avg = self.mle_average(self.particles, probs)      
            pose_avg = mle_avg[0]
            cov = mle_avg[1]
            self.publish_average_pose(pose_avg) 
   
    def laser_callback(self, msg):
        self.scans = msg.ranges

    def odom_callback(self, msg):
        self.odometry = [msg.twist.twist.linear.x*self.dT, msg.twist.twist.linear.y*self.dT, msg.twist.twist.angular.z*self.dT]
   
    def pose_callback(self, msg):
        # msg_frame_pos = msg.pose.pose.position
        # msg_frame_pos = [msg_frame_pos.x, msg_frame_pos.y, msg_frame_pos.z]

        # msg_frame_quat = msg.pose.pose.orientation
        # msg_frame_quat = [msg_frame_quat.x, msg_frame_quat.y,
        #                     msg_frame_quat.z, msg_frame_quat.w]
        # (roll, pitch, yaw) = euler_from_quaternion(msg_frame_quat)

        # x = msg_frame_pos[0]+np.cos(yaw)*msg.point.x-np.sin(yaw)*msg.point.y
        # y = msg_frame_pos[1]+np.cos(yaw)*msg.point.y+np.sin(yaw)*msg.point.x
        # #------------
        # covar = np.reshape(msg.pose.covariance, (6,6))
        # #------------
        # self.particles = np.random.multivariate_normal([x,y], covar, size=(100, 3))

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        angle = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        self.get_logger().info(f"first pose at: {x}, {y}, {angle}")

        def normalize_angle(angle):
            # Normalize the angle to be within the range [-π, π]
            normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
            return normalized_angle
        
        x = np.random.normal(loc=x, scale=1.0, size=(self.num_particles,1))
        y = np.random.normal(loc=y, scale=1.0, size=(self.num_particles,1))
        theta = np.random.normal(loc=angle, scale=1.0, size=(self.num_particles,1))
        theta = np.apply_along_axis(normalize_angle, axis=0, arr=theta)

        self.particles = np.hstack((x, y, theta))
        self.initiated = True

    def mle_average(self, particles, probs, percentile=10):
        N = len(particles)
        num_top = max(1, int(N * percentile / 100))

        # Pair particles with weights and sort
        pw = list(zip(particles, probs))
        pw_sorted = sorted(pw, key=lambda x: x[1], reverse=True)
        top_pw = pw_sorted[:num_top]

        # Extract top particles and weights
        top_particles = [p for p, _ in top_pw]
        top_weights = [w for _, w in top_pw]

        # Normalize weights
        total_weight = sum(top_weights)
        norm_weights = [w / total_weight for w in top_weights]

        # Weighted average
        x_avg = sum(p[0] * w for p, w in zip(top_particles, norm_weights))
        y_avg = sum(p[1] * w for p, w in zip(top_particles, norm_weights))

        # Circular mean for theta
        sin_sum = sum(np.sin(p[2]) * w for p, w in zip(top_particles, norm_weights))
        cos_sum = sum(np.cos(p[2]) * w for p, w in zip(top_particles, norm_weights))
        theta_avg = np.arctan2(sin_sum, cos_sum)

        # Compute covariance
        cov = np.zeros((3, 3))
        for p, w in zip(top_particles, norm_weights):
            dx = p[0] - x_avg
            dy = p[1] - y_avg
            dtheta = np.arctan2(np.sin(p[2] - theta_avg), np.cos(p[2] - theta_avg))  # shortest angular difference

            delta = np.array([dx, dy, dtheta]).reshape((3, 1))
            cov += w * (delta @ delta.T)

        return (x_avg, y_avg, theta_avg), cov
        
    def publish_average_pose(self, pose_avg):

        msg = Odometry()
        msg.header.frame_id = '/map'
        msg.pose.pose.position.x = pose_avg[0]
        msg.pose.pose.position.y = pose_avg[1]

        # z axis rotation
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(pose_avg[2] / 2)
        msg.pose.pose.orientation.w = np.cos(pose_avg[2] / 2)

        msg.child_frame_id = '/base_link'
        self.odom_pub.publish(msg)

        obj = geometry_msgs.msg.TransformStamped()
        obj.header.frame_id = '/map'
        obj.child_frame_id = '/base_link_pf'

        obj.transform.translation.x = pose_avg[0]
        obj.transform.translation.y = pose_avg[1]
        obj.transform.translation.z = 0.0

        obj.transform.rotation.x = 0.0
        obj.transform.rotation.y = 0.0
        obj.transform.rotation.z = np.sin(pose_avg[2] / 2)
        obj.transform.rotation.w = np.cos(pose_avg[2] / 2)

        self.broadcaster.sendTransform(obj)

        try:
            tf_world_to_car: geometry_msgs.msg.TransformStamped = self.buffer.lookup_transform('map', 'base_link',
                                                                                rclpy.time.Time())
            
            x_expected = tf_world_to_car.transform.translation.x
            y_expected = tf_world_to_car.transform.translation.y
            angle_expected = 2 * np.arctan2(tf_world_to_car.transform.rotation.z, tf_world_to_car.transform.rotation.w)

            distance_error_msg = Float32()
            distance_error_msg.data = np.sqrt((x_expected - pose_avg[0])**2 + (y_expected - pose_avg[1])**2)
            
            angle_error_msg = Float32()
            angle_error_msg.data = angle_expected - pose_avg[2]

            self.dist_error_pub.publish(distance_error_msg)
            self.angle_error_pub.publish(angle_error_msg)

        except tf2_ros.TransformException:
            self.get_logger().info('no transform from world to base_link_gt found')
            return

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()