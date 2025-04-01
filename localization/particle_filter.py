from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

from rclpy.node import Node
import numpy as np

from tf_transformations import euler_from_quaternion

from geometry_msgs.msg import PointStamped

assert rclpy


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
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        self.create_subscription(PointStamped,
            "/clicked_point", self.clicked_callback, 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.timer = self.create_timer(self.dT, self.timer_callback)

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

            
    
    def laser_callback(self, msg):
        self.scans = msg.ranges

    def odom_callback(self, msg):
        self.odometry = [msg.twist.twist.linear.x*self.dT, msg.twist.twist.linear.y*self.dT, msg.twist.twist.angular.z*self.dT]

    def clicked_callback(self, msg):
        if not self.initiated:
            # Store clicked point in the map frame
            t = self.tfBuffer.lookup_transform(
                self.message_frame, msg.header.frame_id, rclpy.time.Time())
            
            msg_frame_pos = t.transform.translation
            msg_frame_quat = t.transform.rotation
            msg_frame_quat = [msg_frame_quat.x, msg_frame_quat.y,
                            msg_frame_quat.z, msg_frame_quat.w]
            msg_frame_pos = [msg_frame_pos.x, msg_frame_pos.y, msg_frame_pos.z]
            
            (roll, pitch, yaw) = euler_from_quaternion(msg_frame_quat)

            x = msg_frame_pos[0]+np.cos(yaw)*msg.point.x-np.sin(yaw)*msg.point.y
            y = msg_frame_pos[1]+np.cos(yaw)*msg.point.y+np.sin(yaw)*msg.point.x

            self.particles = np.random.uniform(
                low=[x-0.5, y-0.5, 0], high=[x+0.5, y+0.5, 2*np.pi], size=(100, 3))

            self.initiated = True

    def mle_pose(particles, probs, percentile = 10):
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
        theta_avg = np.atan2(sin_sum, cos_sum)

        return (x_avg, y_avg, theta_avg)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
