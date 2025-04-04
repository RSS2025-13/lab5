#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult
from robot_wall_follower.evaluations import Evaluations
from robot_wall_follower.visualization_tools import VisualizationTools
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime


class WallFollower(Node):

    WALL_TOPIC = "/wall"

    def __init__(self):
        super().__init__("wall_follower")
        # Declare parameters to make them available for use
        # DO NOT MODIFY THIS! 
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", -1)
        self.declare_parameter("velocity", 0.5)
        self.declare_parameter("desired_distance", 0.5)
        self.declare_parameter("kp", 0.8)
        self.declare_parameter("kd", 0.3)
        self.declare_parameter("ki", 0.0)

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
        self.line_pub = self.create_publisher(Marker, self.WALL_TOPIC, 1)

        self.scan_subscriber = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.listener_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)

        self.previous_error = 0
        self.previous_time = self.get_clock().now()
        self.left_or_right = self.SIDE

        self.kp = self.get_parameter('kp').get_parameter_value().double_value
        self.kd = self.get_parameter('kd').get_parameter_value().double_value
        self.ki = self.get_parameter('ki').get_parameter_value().double_value

        self.get_logger().info(f'kp: "{self.kp}"')
        self.get_logger().info(f'kd: "{self.kd}"')
        self.get_logger().info(f'v2')

        self.evals = Evaluations()

        # Add these new attributes for plotting
        self.distances = []
        self.times = []
        self.start_time = self.get_clock().now()
        
        # Create output directory if it doesn't exist
        self.output_dir = os.path.expanduser('~/wall_follower_plots')
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # TODO: Write your callback functions here

    def listener_callback(self, msg):
        ads_msg = AckermannDriveStamped()

        now = self.get_clock().now()

        ads_msg.header.stamp = now.to_msg()
        ads_msg.header.frame_id = 'base_link'
        
        
        ads_msg.drive.speed = self.VELOCITY
        #ads_msg.drive.steering_angle = np.pi/4
        ads_msg.drive.steering_angle = self.PID_angle(laser_msg = msg, current_time = now)

        self.drive_publisher.publish(ads_msg)

        #self.get_logger().info(f'min angle: "{msg.angle_min}" and max_angle: "{msg.angle_max}"')
        #self.get_logger().info(f'published angle "{ads_msg.drive.steering_angle}"and speed "{ads_msg.drive.speed}"')

    def PID_angle(self, laser_msg, current_time):
        #Creating array of desired values
        ranges_array = np.array(laser_msg.ranges)   #converts ranges array to numpy array for manipulation
        min_angle_sweep = -3*np.pi/4   #-2*np.pi/3
        max_angle_sweep =  np.pi/12 #-np.pi/6  #Gets the max angle to sweep to (min is -3pi/4)
        
        number_of_elements_sweeped = int((max_angle_sweep - min_angle_sweep) / laser_msg.angle_increment) + 1   #determines the number of values within that sweep given the angle increment
        first_index = int(np.abs(laser_msg.angle_min - min_angle_sweep) / laser_msg.angle_increment)
        last_index = first_index + number_of_elements_sweeped
        side_ranges_array = ranges_array[:last_index] if self.SIDE == -1 else ranges_array[-last_index:]     #Creates an array of just the desired values from that side

        #creates x and y values from each Lidar Scan point
        thetas = np.arange(min_angle_sweep, max_angle_sweep, laser_msg.angle_increment) if self.SIDE == -1 else np.arange(-1 * max_angle_sweep, -1 * min_angle_sweep, laser_msg.angle_increment)
        x_array = side_ranges_array * np.cos(thetas)
        y_array = side_ranges_array * np.sin(thetas)

        #create a LSR 

        #limiting values with r > 4m
        mask = side_ranges_array <= 4  # Boolean mask where values are <= 4
        new_x_array = x_array[mask]
        new_y_array = y_array[mask]

        coeffs = np.polyfit(new_x_array, new_y_array, 1)

        #Find the shortest distance between the line and the car
        d = np.abs(coeffs[1]) / np.sqrt(1 + (coeffs[0]**2))

        # Add data points for plotting with limit
        elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
        self.times.append(elapsed_time)
        self.distances.append(d)

        #distance to front wall
        d_front = ranges_array[len(ranges_array)//2]

        #distance to side wall
        side_index = (np.abs(thetas - (self.SIDE*np.pi/3)).argmin())
        d_side = side_ranges_array[side_index]

        if d_front < 1.5*(self.VELOCITY/0.5):
            factor = d_front/3
            d  = d * factor

        if d_side > 5.0:
            factor = d_side/3
            d = d*factor
        
        #update evals
        self.evals.update(d, self.DESIRED_DISTANCE)

        #Determine error and PID controller
        error = (self.DESIRED_DISTANCE - d) * (-1 if self.SIDE == 1 else 1)

        #self.left_or_right = (-1 if self.previous_error <= error else 1) * self.left_or_right
        delta_e = error - self.previous_error
        delta_t = (current_time - self.previous_time).nanoseconds / 1e9

        proportional = self.kp * error

        derivative = self.kd * (delta_e / delta_t)

        integral = self.ki * (error * delta_t)

        u_t = (proportional + derivative + integral) #* self.left_or_right

        self.previous_time = current_time
        self.previous_error = error

        #Visualization
        y_visual = coeffs[0] * new_x_array + coeffs[1]
        VisualizationTools.plot_line(new_x_array, y_visual, self.line_pub, frame="/laser")

        return u_t
    
    
    def parameters_callback(self, params):
        """
        This is used by the test cases to modify the parameters during testing. 
        It's called whenever a parameter is set via 'ros2 param set'.
        """
        # Save current plot before parameter change
        if hasattr(self, 'times') and len(self.times) > 0:
            self.save_plot()

        # Reset plotting data
        self.distances = []
        self.times = []
        self.start_time = self.get_clock().now()

        # Update parameters
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
            elif param.name == 'kp':
                self.kp = param.value
                self.get_logger().info(f"Updated kp to {self.kp}")
            elif param.name == 'kd':
                self.kd = param.value
                self.get_logger().info(f"Updated kd to {self.kd}")
            elif param.name == 'ki':
                self.ki = param.value
                self.get_logger().info(f"Updated ki to {self.ki}")
        return SetParametersResult(successful=True)

    def save_plot(self):
        """Create and save the distance plot"""
        try:
            sum = 0
            for i in self.distances:
                sum += np.abs(i - self.DESIRED_DISTANCE)
            error = sum/len(self.times)

            # Add error text annotation to the plot
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.times, self.distances, label='Actual Distance', color='blue')
            plt.axhline(y=self.DESIRED_DISTANCE, color='r', linestyle='--', label='Desired Distance')
            
            plt.xlabel('Time (seconds)')
            plt.ylabel('Distance (meters)')
            plt.title('Wall Following Distance Over Time')
            error_text = f'Average Error: {error:.3f} m'
            plt.text(0.02, 0.98, error_text, 
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.legend()
            plt.grid(True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'wall_follower_plot_{timestamp}_kp_{str(self.kp).replace(".", "_")}_kd_{str(self.kd).replace(".", "_")}_ki_{str(self.ki).replace(".", "_")}_v_{str(self.VELOCITY).replace(".", "_")}_d_{str(self.DESIRED_DISTANCE).replace(".", "_")}.png'
            filepath = os.path.expanduser("~/racecar_ws/src/wall_follower/graphs/")
            Path(filepath).mkdir(parents=True, exist_ok=True)
            full_path = os.path.join(filepath, filename)
            
            plt.savefig(full_path)
            plt.close()  # Close the figure to free memory
            self.get_logger().info(f'Saved plot to {full_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save plot: {str(e)}')


def main():
    rclpy.init()
    wall_follower = WallFollower()
    rclpy.spin(wall_follower)
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    