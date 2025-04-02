#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
#from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult

#from wall_follower.evaluations import Evaluations

#from wall_follower.visualization_tools import VisualizationTools


class WallFollower(Node):

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS! 
        self.declare_parameter("scan_topic", "default")
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("side", -1)
        self.declare_parameter("velocity", 1.0)
        self.declare_parameter("desired_distance", 1.0)

        # Fetch constants from the ROS parameter server
        # DO NOT MODIFY THIS! This is necessary for the tests to be able to test varying parameters!
        self.SCAN_TOPIC = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.SIDE = self.get_parameter('side').get_parameter_value().integer_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.DESIRED_DISTANCE = self.get_parameter('desired_distance').get_parameter_value().double_value

         # This activates the parameters_callback function so that the tests are able
        # to change the parameters during testing.
        # DO NOT MODIFY THIS! 
        self.add_on_set_parameters_callback(self.parameters_callback)
		
	# TODO: Initialize your publishers and subscribers here
        self.scan_subscriber = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 10) 
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        # a publisher for our line marker
        #self.line_pub = self.create_publisher(Marker, '/wall', 1)

        #self.evals = Evaluations()

    # TODO: Write your callback functions here    
    def scan_callback(self, msg):
        # Get the distances
        msg_ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg_ranges))
        self.angle_min = np.pi/16
        self.angle_max = np.pi/4
        self.angle_front = np.pi/64

        #Create a dedicated range for the front of the vehicle
        lower_index = (np.abs(angles - (-1*self.angle_front)).argmin())
        upper_index = (np.abs(angles - (self.angle_front)).argmin())
        front_distances = msg_ranges[lower_index:upper_index]

        # Isolate the ranges that you are looking at depending on the side
        if self.SIDE == -1:
            lower_index = (np.abs(angles - (-1*self.angle_max)).argmin())
            upper_index = (np.abs(angles - (-1*self.angle_min)).argmin())
            distances = msg_ranges[lower_index:upper_index]
        else:
            lower_index = (np.abs(angles - (self.angle_min)).argmin())
            upper_index = (np.abs(angles - (self.angle_max)).argmin())
            distances = msg_ranges[lower_index:upper_index]
        
        # Check if we have valid distances
        if len(distances) == 0:
            self.get_logger().warn("No valid distances found in range!")
            return
        #self.get_logger().info(f'distances: {distances}')
        # Set Kp and Kd values
        self.Kp = 1.0
        self.Kd = 0.5

        # Initialize previous_error if not already set
        if not hasattr(self, 'previous_error'):
            self.previous_error = 0.0
    
        # Use linear regression to fit a line to the distances
        slope, intercept = self.linear_regression(distances)

        # Calculate the perpendicular distance to the line
        distance = self.perpendicular_distance(slope, intercept)

        if np.any(front_distances < 1.5):
            distancef = np.mean(front_distances)
            factor = distancef/3
            distance  = distance * factor

        # Calculate the Error
        error = self.DESIRED_DISTANCE - distance

        # Calculate the Derivative
        derivative = error - self.previous_error
        self.previous_error = error  # Save error for next iteration

        # Calculate the Steering Angle
        steering_angle = self.Kp * error + self.Kd * derivative
        #self.get_logger().info(f'{steering_angle}')

        # Create an AckermannDriveStamped message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = (-1*self.SIDE) * steering_angle
        drive_msg.drive.acceleration = 0.0
        drive_msg.drive.jerk = 0.0
        drive_msg.drive.speed = self.VELOCITY

        # Publish the drive message
        self.drive_publisher.publish(drive_msg)

        #update evals
        #self.evals.update(distance)
        
    
    def linear_regression(self, distances):
        # Convert polar coordinates to Cartesian
        angles = np.linspace(-1*self.angle_max, -1*self.angle_min, len(distances)) if self.SIDE == -1 else np.linspace(self.angle_min, self.angle_max, len(distances))
        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        
        # Fit line to Cartesian points
        slope, intercept = np.polyfit(x, y, 1)

        # Visualize the line
        # Create points along the fitted line for visualization
        x_line = np.array([np.min(x), np.max(x)])  # Use range of detected points
        y_line = slope * x_line + intercept  # Calculate corresponding y values
        #VisualizationTools.plot_line(x_line, y_line, self.line_pub)

        return slope, intercept

    def perpendicular_distance(self, slope, intercept):
        #Calculate the perpendicular distance to the line
        distance = np.abs(intercept) / np.sqrt(slope**2 + 1)
        return distance

    def parameters_callback(self, params):
        """
        DO NOT MODIFY THIS CALLBACK FUNCTION!
        
        This is used by the test cases to modify the parameters during testing. 
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        for param in params:
            if param.name == 'side':
                self.SIDE = param.value
                self.get_logger().info(f"Updated side to {self.SIDE}")
            elif param.name == 'velocity':
                self.VELOCITY = param.value
                self.get_logger().info(f"Updated velocity to {self.VELOCITY}")
            elif param.name == 'desired_distance':
                self.DESIRED_DISTANCE = param.value
                self.get_logger().info(f"Updated desired_distance to {self.DESIRED_DISTANCE}")
        return SetParametersResult(successful=True)


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()