#!/usr/bin/env python3.8
import sys
import os

# Add devel/lib to Python path for pycloud_matcher module
devel_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', '..', 'devel', 'lib')
if os.path.exists(devel_lib):
    sys.path.insert(0, devel_lib)

import rospy
import zlib
import numpy as np
import re
from geometry_msgs.msg import PoseStamped
# from fast_lio.msg import RelocalizationMsg
from std_msgs.msg import UInt8MultiArray
from std_srvs.srv import Trigger
from tf.transformations import quaternion_from_euler, quaternion_matrix, euler_from_matrix


class LocalizationInitPosePublisher:
    def __init__(self):
        rospy.init_node('localization_init_pose_pub')

        # Member variables
        self.robot_num = rospy.get_param("common/robot_num", 1)
        init_pose_list = rospy.get_param("common/init_pose", [0,0,0,0,0,0])
        # Create robot_num x 6 array for init poses (x,y,z,roll,pitch,yaw)
        self.init_pose = np.array(init_pose_list).reshape(self.robot_num, 6)
        self.prefix = rospy.get_namespace().replace('/', '')
        self.cloud_received = [False]*self.robot_num
        self.point_cloud_data = [None]*self.robot_num
        self.sub_cloud = []
        self.pub_init_pose = []
        for i in range(self.robot_num):
            if len(self.prefix) > 0: topic_name = f'/{self.prefix[:-1]}{i+1}/init_pose'
            else: topic_name = 'init_pose'
            self.pub_init_pose.append(rospy.Publisher(topic_name, PoseStamped, queue_size=1, latch=True))

        # Numpy dtype for PointXYZINormal (32 bytes per point)
        self.cloud_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
            ('normal_x', np.float32),
            ('normal_y', np.float32),
            ('normal_z', np.float32),
            ('curvature', np.float32),
        ])

    def decompress_cloud(self, compressed_data):
        """Decompress zlib data."""
        try:
            decompressed = zlib.decompress(bytes(compressed_data))
            return decompressed
        except zlib.error as e:
            rospy.logerr(f"Decompress error: {e}")
            return None

    def parse_cloud(self, data):
        """Parse byte data to numpy array (PointXYZINormal)."""
        try:
            # Read as raw float32 array (8 floats per point)
            float_array = np.frombuffer(data, dtype=np.float32)

            # Check if we have complete points (8 floats each)
            if float_array.size % 8 != 0:
                rospy.logwarn(f"Data has {float_array.size} floats, not multiple of 8. Truncating...")
                float_array = float_array[:float_array.size - (float_array.size % 8)]

            # Reshape to N x 8
            n_points = float_array.size // 8
            float_array = float_array.reshape(n_points, 8)

            # Convert to structured array
            points = np.zeros(n_points, dtype=self.cloud_dtype)
            points['x'] = float_array[:, 0]
            points['y'] = float_array[:, 1]
            points['z'] = float_array[:, 2]
            points['intensity'] = float_array[:, 3]
            points['normal_x'] = float_array[:, 4]
            points['normal_y'] = float_array[:, 5]
            points['normal_z'] = float_array[:, 6]
            points['curvature'] = float_array[:, 7]

            return points
        except Exception as e:
            rospy.logerr(f"Parse error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None

    def cloud_callback(self, msg, topic_name):
        """Callback for compressed cloud message."""
        # Extract robot ID from topic name (e.g., "/robot1/cloud_raw_compressed" -> "1")
        match = re.search(r'\d+', topic_name)
        robot_id = int(match.group()) if match else 0

        rospy.loginfo(f"Received from topic: {topic_name}, robot_id: {robot_id}")

        # Decompress
        decompressed = self.decompress_cloud(msg.data)
        if decompressed is None: return

        # Parse to point cloud
        points = self.parse_cloud(decompressed)
        if points is None: return

        self.point_cloud_data[robot_id-1] = points
        self.cloud_received[robot_id-1] = True

    def convert_structured_to_float_array(self, points_structured):
        """Convert structured numpy array to float32 array for pycloud_matcher."""
        # Extract xyz and normal fields as N x 8 float32 array
        n_points = len(points_structured)
        float_array = np.zeros((n_points, 8), dtype=np.float32)
        float_array[:, 0] = points_structured['x']
        float_array[:, 1] = points_structured['y']
        float_array[:, 2] = points_structured['z']
        float_array[:, 3] = points_structured['intensity']
        float_array[:, 4] = points_structured['normal_x']
        float_array[:, 5] = points_structured['normal_y']
        float_array[:, 6] = points_structured['normal_z']
        float_array[:, 7] = points_structured['curvature']
        return float_array

    def perform_ndt_matching(self):
        """Perform NDT matching for all point clouds to the first one."""
        try:
            import pycloud_matcher
        except ImportError:
            rospy.logerr("pycloud_matcher module not found. Please build the project with pybind11.")
            return None

        rospy.loginfo("Starting NDT matching for all robots...")

        # Convert all point clouds to float32 arrays
        cloud_list = []
        for i, points in enumerate(self.point_cloud_data):
            if points is None:
                rospy.logwarn(f"Robot {i+1} point cloud is None, skipping...")
                continue
            cloud_list.append(self.convert_structured_to_float_array(points))
            rospy.loginfo(f"Robot {i+1}: {len(points)} points")

        if len(cloud_list) < 2:
            rospy.logwarn(f"Need at least 2 point clouds for NDT matching, got {len(cloud_list)}")
            return None

        # Call batch NDT matching
        try:
            results = pycloud_matcher.batch_match(
                cloud_list,
                self.init_pose,  # Initial poses (N x 6)
                max_correspondence_distance=0.5,  # Maximum correspondence distance for GICP
                trans_epsilon=1e-6,
                max_iterations=1000,
                downsample_leaf_size=0.0
            )
        except Exception as e:
            rospy.logerr(f"NDT matching failed: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None

        # Process results
        relative_poses = []
        for i, res in enumerate(results):
            transform = res[0]  # 4x4 transformation matrix
            rospy.loginfo(f"Robot {i+1}: converged={res[1]}, fitness={res[2]:.4f}")

            relative_poses.append(transform)

        rospy.loginfo("NDT matching completed!")
        return relative_poses

    def request_point_cloud(self, service_name, timeout=5.0):
        """Request point cloud publish via service."""
        rospy.loginfo(f"Waiting for service {service_name}...")
        try:
            rospy.wait_for_service(service_name, timeout=timeout)
        except rospy.ROSException:
            rospy.logerr(f"Service {service_name} not available!")
            return False

        try:
            proxy = rospy.ServiceProxy(service_name, Trigger)
            resp = proxy()
            if resp.success:
                rospy.loginfo(f"Service call success: {resp.message}")
                return True
            else:
                rospy.logwarn(f"Service call failed: {resp.message}")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def publish_relocalization_msg(self, x, y, z, roll, pitch, yaw, robot_id=0):
        """Publish relocalization message."""
        init_pose = PoseStamped()
        init_pose.header.frame_id = "map"
        init_pose.header.stamp = rospy.Time.now()

        init_pose.pose.position.x = x
        init_pose.pose.position.y = y
        init_pose.pose.position.z = z

        quaternion = quaternion_from_euler(roll, pitch, yaw)
        init_pose.pose.orientation.x = quaternion[0]
        init_pose.pose.orientation.y = quaternion[1]
        init_pose.pose.orientation.z = quaternion[2]
        init_pose.pose.orientation.w = quaternion[3]

        self.pub_init_pose[robot_id].publish(init_pose)

    def run(self):
        """Main run loop."""      
        # Request point cloud from each robot
        # if len(self.prefix) > 0:  # Multiple robots
        for i in range(self.robot_num):
            if len(self.prefix) > 0: 
                topic_name = f'/{self.prefix[:-1]}{i+1}/cloud_raw_compressed'
                service_name = f'/{self.prefix[:-1]}{i+1}/publish_cloud'
            else: 
                topic_name = 'cloud_raw_compressed'
                service_name = 'publish_cloud'
            self.sub_cloud.append(
                rospy.Subscriber(topic_name, UInt8MultiArray, self.cloud_callback, callback_args=topic_name)
            )            
            # rospy.loginfo(f"Requesting point cloud from {service_name}...")
            self.request_point_cloud(service_name, timeout=30)

        while not rospy.is_shutdown():
            if all(self.cloud_received):
                rospy.loginfo("All point clouds received, performing NDT matching...")
                rel_pose = self.perform_ndt_matching() # Relative poses for all robots to robot 1
                for i in range(self.robot_num):
                    if i==0:
                        # Robot 1: use init_pose directly
                        x, y, z, roll, pitch, yaw = self.init_pose[i]
                    else:
                        # Other robots: result_T = init_T @ rel_pose[i]
                        transform = rel_pose[i]
                        init_T = np.eye(4)
                        init_T[0,3] = self.init_pose[0][0]
                        init_T[1,3] = self.init_pose[0][1]
                        init_T[2,3] = self.init_pose[0][2]
                        init_quat = quaternion_from_euler(self.init_pose[0][3], self.init_pose[0][4], self.init_pose[0][5])
                        init_T[:3,:3] = quaternion_matrix(init_quat)[:3,:3]
                        result_T = init_T @ transform
                        x = result_T[0, 3]
                        y = result_T[1, 3]
                        z = result_T[2, 3]
                        # Extract rotation (roll, pitch, yaw) from result
                        roll, pitch, yaw = euler_from_matrix(result_T)
                    self.publish_relocalization_msg(x, y, z, roll, pitch, yaw, robot_id=i)
                break
            rospy.sleep(0.2)


if __name__ == '__main__':
    node = LocalizationInitPosePublisher()
    node.run()
