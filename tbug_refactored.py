import rclpy
import math
import numpy as np
import sys
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan

D_REF = 1.0     # desired wall distance
K_P = 1.2        # wall distance gain
V = 0.4          # forward speed
BIAS = 0.6       # wall following bias

class TBug(Node):
    def __init__(self, args):
        super().__init__('tbug_controller')

        # Goal
        self.goal = [float(args[0]), float(args[1])]
        self.get_logger().info(f"Goal: {self.goal}")

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # Lidar & obstacle data
        self.lidar_ranges = []
        self.obstacle_coords = []
        self.obstacle_clusters = {}

        # State machine
        self.state = 'SEEK_GOAL'  # SEEK_GOAL / GO_TO_TANGENT / FOLLOW_WALL
        self.action = 'ROTATE'    # ROTATE / DRIVE
        self.wall_side = None
        self.dmin = np.inf  # min distance to goal while following wall
        self.best_tangent_point = None
        self.tangent_reached = False

        # Parameters
        self.GOAL_REACHED_DIST = 0.5
        self.TANGENT_REACHED_DIST = 1.5
        self.ANGLE_REACHED = math.radians(2)
        self.CLUSTER_THRESHOLD = 1.2
        self.TANGENT_OFFSET = 0.5

        # ROS
        self.create_subscription(LaserScan, '/bcr_bot/scan', self.lidar_callback, 10)
        self.create_subscription(Odometry, '/bcr_bot/odom', self.odom_callback, 10)
        self.vel_pub = self.create_publisher(Twist, '/bcr_bot/cmd_vel', 10)
        self.create_timer(0.1, self.main_loop)

        self.get_logger().info("TBug Controller Started")

    # ===== CALLBACKS =====
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_ranges[self.lidar_ranges == 0.0] = np.inf
        self.lidar_ranges[self.lidar_ranges > msg.range_max] = np.inf
        self.update_obstacles()

    # ===== HELPER FUNCTIONS =====
    def distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def line_of_sight_clear(self):
        if len(self.obstacle_coords) == 0:
            return True

        robot = np.array([self.x, self.y])
        goal = np.array(self.goal)

        for ox, oy, theta in np.array(self.obstacle_coords):
            obs_vec = np.array([ox, oy]) - robot
            goal_vec = goal - robot
            proj = np.dot(obs_vec, goal_vec) / np.linalg.norm(goal_vec)
            if 0 < proj < np.linalg.norm(goal_vec):
                # perpendicular distance to line
                perp = np.linalg.norm(obs_vec - proj * goal_vec / np.linalg.norm(goal_vec))
                if perp < 0.5:
                    return False
        return True

    def update_obstacles(self):
        self.obstacle_coords = []
        self.obstacle_clusters = {}
        
        if not hasattr(self, 'lidar_ranges') or len(self.lidar_ranges) == 0:
            return
            
        prev_x = None
        prev_y = None
        cluster_id = 0
        
        for i, dist in enumerate(self.lidar_ranges):
            if dist < 10.0:
                angle = math.radians(i)
                x = self.x + dist * math.cos(self.yaw + angle)
                y = self.y + dist * math.sin(self.yaw + angle)
                
                # Clustering
                if prev_x is not None and prev_y is not None:
                    if self.distance([x, y], [prev_x, prev_y]) > self.CLUSTER_THRESHOLD:
                        cluster_id += 1
                
                self.obstacle_coords.append([x, y, angle])
                
                if cluster_id not in self.obstacle_clusters:
                    self.obstacle_clusters[cluster_id] = []
                self.obstacle_clusters[cluster_id].append((x, y))
                
                prev_x = x
                prev_y = y 

    def get_tangent_points(self):
        """Extract left and right tangent points from each obstacle cluster"""
        tangent_points = []
        for c, points in self.obstacle_clusters.items():
            if len(points) >= 2:
                left_tangent = points[-1]
                right_tangent = points[0]
                tangent_points.append((c, left_tangent, right_tangent))
        return tangent_points

    def find_best_tangent(self):
        """Find the best tangent point to approach based on minimum cost"""
        tangent_points = self.get_tangent_points()
        
        if len(tangent_points) == 0:
            return None, None
        
        best_distance = float('inf')
        best_tangent = None
        best_side = None
        
        robot_pos = [self.x, self.y]
        
        for i, left, right in tangent_points:
            for pt, side in [(left, 'LEFT'), (right, 'RIGHT')]:
                # Cost = distance to tangent + distance from tangent to goal
                cost = self.distance(robot_pos, pt) + self.distance(pt, self.goal)
                if cost < best_distance:
                    best_distance = cost
                    best_tangent = pt
                    best_side = side
        
        return best_tangent, best_side

    def push_tangent_perpendicular(self, tangent_point, side, offset):
        """Offset tangent point perpendicular to robot-tangent line"""
        rx, ry = self.x, self.y
        tx, ty = tangent_point
        
        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        
        if dist == 0:
            return tangent_point
        
        if side == 'LEFT':
            # Left perpendicular
            nx, ny = -dy / dist, dx / dist
        else:
            # Right perpendicular
            nx, ny = dy / dist, -dx / dist
        
        pushed_x = tx + offset * nx
        pushed_y = ty + offset * ny
        
        return (pushed_x, pushed_y)

    def rotate_to(self, target):
        target_angle = math.atan2(target[1] - self.y, target[0] - self.x)
        error = target_angle - self.yaw
        error = (error + math.pi) % (2 * math.pi) - math.pi

        if abs(error) < self.ANGLE_REACHED:
            self.publish_velocity(0.0, 0.0)
            return True

        angular = np.clip(1.5 * error, -2.0, 2.0)
        self.publish_velocity(0.0, angular)
        return False

    def drive_to(self, target, threshold=None):
        if threshold is None:
            threshold = self.GOAL_REACHED_DIST
            
        dist = self.distance([self.x, self.y], target)
        if dist < threshold:
            self.publish_velocity(0.0, 0.0)
            return True
        self.publish_velocity(min(0.5 * dist, 2.0), 0.0)
        return False

    def publish_velocity(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.vel_pub.publish(msg)

    def wall_follow_right(self):
        r = self.lidar_ranges
        right = np.min(r[   ]])
        

    def wall_follow_left(self):
        r = self.lidar_ranges
        left = np.min(r[80:100])
        wall_angle = self.estimate_wall_angle('LEFT')
        if wall_angle is None:
            self.publish_velocity(0.0, -0.5)
            return

        desired_yaw = wall_angle - math.pi / 2
        angle_error = desired_yaw - self.yaw
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        dist_error = D_REF - left

        angular = 1.2 * angle_error - 0.8 * dist_error
        self.publish_velocity(V, np.clip(angular, -1.2, 1.2))


    def main_loop(self):
        dist_to_goal = self.distance([self.x, self.y], self.goal)
        if dist_to_goal < self.GOAL_REACHED_DIST:
            self.get_logger().warn("GOAL REACHED!")
            self.publish_velocity(0.0, 0.0)
            return

        if self.state == 'SEEK_GOAL':
            self.get_logger().info(f"SEEK_GOAL | dist={dist_to_goal:.2f}")
            if self.line_of_sight_clear():
                target = self.goal
                if self.action == 'ROTATE' and self.rotate_to(target):
                    self.action = 'DRIVE'
                elif self.action == 'DRIVE' and self.drive_to(target):
                    self.action = 'ROTATE'
            else:
                # Blocked → find tangent point
                self.get_logger().info("Goal blocked → Finding tangent point")
                tangent, side = self.find_best_tangent()
                
                if tangent is not None:
                    self.best_tangent_point = self.push_tangent_perpendicular(
                        tangent, side, self.TANGENT_OFFSET
                    )
                    self.wall_side = side
                    self.dmin = dist_to_goal
                    self.state = 'GO_TO_TANGENT'
                    self.action = 'ROTATE'
                    self.tangent_reached = False
                    self.get_logger().info(f"Going to tangent point: {self.best_tangent_point}")
                else:
                    # No tangent found, try wall following directly
                    self.wall_side = 'LEFT'
                    self.dmin = dist_to_goal
                    self.state = 'FOLLOW_WALL'
                    self.action = 'ROTATE'

        elif self.state == 'GO_TO_TANGENT':
            self.get_logger().info(f"GO_TO_TANGENT | target={self.best_tangent_point}")
            
            # Check if we can see goal now
            if self.line_of_sight_clear() and dist_to_goal < self.dmin:
                self.get_logger().info("LOS clear from tangent approach → SEEK_GOAL")
                self.state = 'SEEK_GOAL'
                self.action = 'ROTATE'
                self.best_tangent_point = None
                return
            
            # Navigate to tangent point
            if self.action == 'ROTATE' and self.rotate_to(self.best_tangent_point):
                self.action = 'DRIVE'
            elif self.action == 'DRIVE':
                if self.drive_to(self.best_tangent_point, self.TANGENT_REACHED_DIST):
                    self.get_logger().info("Tangent point reached → FOLLOW_WALL")
                    self.state = 'FOLLOW_WALL'
                    self.action = 'ROTATE'
                    self.tangent_reached = True
                    self.best_tangent_point = None

        elif self.state == 'FOLLOW_WALL':
            self.get_logger().info(f"FOLLOW_WALL ({self.wall_side}) | dmin={self.dmin:.2f}")
            
            # Update dmin
            if dist_to_goal < self.dmin:
                self.dmin = dist_to_goal
            
            # Check if we can leave wall following
            if self.line_of_sight_clear() and dist_to_goal < self.dmin:
                self.get_logger().info("Exit wall-follow → SEEK_GOAL")
                self.state = 'SEEK_GOAL'
                self.action = 'ROTATE'
                self.wall_side = None
                return
            
            # Do wall following
            if self.wall_side == 'RIGHT':
                self.wall_follow_right()
            else:
                self.wall_follow_left()




def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 3:
        print("Usage: python3 tbug.py <goal_x> <goal_y>")
        sys.exit(1)
    node = TBug(sys.argv[1:3])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
