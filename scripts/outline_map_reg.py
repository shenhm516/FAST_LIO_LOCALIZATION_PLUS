#!/usr/bin/env python3
"""
Point cloud coarse registration using FPFH + RANSAC and ICP refinement.
FPFH with relative radius is robust to varying point cloud density.
"""
import open3d as o3d
import numpy as np
import os
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_matrix


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def load_pcd_files(pcd_dir, source_file, target_file):
    """Load source and target point clouds from PCD directory."""
    source_path = os.path.join(pcd_dir, source_file)
    target_path = os.path.join(pcd_dir, target_file)

    print(f"Loading source point cloud: {source_path}")
    source = o3d.io.read_point_cloud(source_path)
    print(f"  Source points: {len(source.points)}")

    print(f"Loading target point cloud: {target_path}")
    target = o3d.io.read_point_cloud(target_path)
    print(f"  Target points: {len(target.points)}")

    return source, target


def preprocess_point_cloud(pcd, voxel_size=0.5, downsample=True):
    """
    Preprocess point cloud: downsample and compute FPFH features.

    FPFH with relative radius is robust to varying point cloud density.

    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for downsampling (used as reference for feature radius)
        downsample: Whether to perform downsampling (default: True)

    Returns:
        pcd_down: Downsampled point cloud (or original if downsample=False)
        pcd_fpfh: FPFH features
    """
    # Downsample if enabled
    if downsample:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = pcd

    # Compute normals with adaptive search
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=30)
    )

    # Compute FPFH features with larger relative radius for density invariance
    # Using relative radius (voxel_size * multiplier) makes FPFH robust to density changes
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 4, max_nn=100)
    )

    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, source_fpfh, target_down, target_fpfh,
                                voxel_size):
    """
    Execute global registration using RANSAC.
    """
    distance_threshold = voxel_size * 1.5

    print("Running RANSAC global registration...")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result


def execute_icp_refinement(source, target, initial_transform, voxel_size):
    """
    Refine registration using multi-stage ICP with coarse-to-fine strategy.

    The distance threshold decreases gradually to avoid local minima:
    - Stage 1: voxel_size * 5.0 (coarse matching)
    - Stage 2: voxel_size * 2.0
    - Stage 3: voxel_size * 1.0
    - Stage 4: voxel_size * 0.5 (fine matching)
    """
    # Define coarse-to-fine distance thresholds
    distance_thresholds = [voxel_size * 5.0, voxel_size * 2.0,
                           voxel_size * 1, voxel_size * 0.5]
    max_iterations = [500, 500, 1000, 2000]  # More iterations for finer stages

    current_transform = initial_transform

    print("Running coarse-to-fine ICP refinement...")
    for i, (dist_thresh, max_iter) in enumerate(zip(distance_thresholds, max_iterations)):
        print(f"  Stage {i+1}: distance_threshold={dist_thresh:.3f}, max_iter={max_iter}")

        result = o3d.pipelines.registration.registration_icp(
            source, target, dist_thresh, current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
        )

        print(f"    Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")

        # Use this stage's result as initialization for next stage
        current_transform = result.transformation

    return result


def visualize_registration(source, target, transformation=None):
    """
    Visualize the registration result.
    """
    source_temp = source.clone()
    target_temp = target.clone()

    # Color the point clouds
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue

    if transformation is not None:
        source_temp.transform(transformation)

    o3d.visualization.draw_geometries([source_temp, target_temp])


def point_cloud2_to_open3d(cloud_msg):
    """Convert sensor_msgs/PointCloud2 to Open3D point cloud."""
    gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(gen))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def coarse_registration_with_pcd(source_pcd, target_pcd, voxel_size=0.5):
    """
    Perform coarse-to-fine registration: FPFH + RANSAC -> ICP Refine.

    Args:
        source_pcd: Source Open3D point cloud
        target_pcd: Target Open3D point cloud
        voxel_size: Voxel size for downsampling

    Returns:
        transformation: 4x4 transformation matrix
        fitness: Registration fitness score
        rmse: Registration RMSE
    """
    print("\n" + "="*60)
    print("FPFH + RANSAC -> ICP Registration")
    print("="*60)

    # Preprocess
    print(f"\nPreprocessing point clouds...")
    print(f"  Source: no downsampling")
    print(f"  Target: voxel_size={voxel_size:.3f}")
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size, downsample=False)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size, downsample=True)

    print(f"Source points: {len(source_down.points)}")
    print(f"Target points: {len(target_down.points)}")

    # Global registration using RANSAC
    print("\n=== Global Registration (RANSAC) ===")
    result_ransac = execute_global_registration(
        source_down, source_fpfh, target_down, target_fpfh, voxel_size
    )

    print(f"RANSAC Result:")
    print(f"  Fitness: {result_ransac.fitness:.4f}")
    print(f"  RMSE: {result_ransac.inlier_rmse:.4f}")

    # ICP refinement
    print("\n=== Local Registration (ICP) ===")
    result_icp = execute_icp_refinement(
        source_down, target_down, result_ransac.transformation, voxel_size
    )

    print(f"ICP Result:")
    print(f"  Fitness: {result_icp.fitness:.4f}")
    print(f"  RMSE: {result_icp.inlier_rmse:.4f}")
    print(f"  Transformation:\n{result_icp.transformation}")

    print("\n" + "="*60)
    print("Registration completed!")
    print(f"Final fitness: {result_icp.fitness:.4f}")
    print(f"Final RMSE: {result_icp.inlier_rmse:.4f}")
    print("="*60 + "\n")

    return result_icp.transformation, result_icp.fitness, result_icp.inlier_rmse


class PointCloudAccumulationNode:
    """ROS node for accumulating point clouds and performing registration."""

    def __init__(self, target_pcd_path, voxel_size=0.5, num_frames=5,
                 input_topic="/velodyne_points", pcd_dir=None):
        """
        Initialize the accumulation node.

        Args:
            target_pcd_path: Path to the target map PCD file (e.g., aero_maze.pcd)
            voxel_size: Voxel size for downsampling
            num_frames: Number of frames to accumulate before registration
            input_topic: ROS topic to subscribe to for point cloud input
            pcd_dir: Directory containing PCD files (for saving results)
        """
        rospy.init_node('point_cloud_accumulation', anonymous=True)

        self.target_pcd_path = target_pcd_path
        self.voxel_size = voxel_size
        self.num_frames = num_frames
        self.input_topic = input_topic
        self.pcd_dir = pcd_dir if pcd_dir else os.path.join(get_script_dir(), "../PCD")

        # Accumulation buffer
        self.accumulated_points = []
        self.frame_count = 0
        self.registration_done = False

        # Load target map
        if os.path.exists(self.target_pcd_path):
            self.target_pcd = o3d.io.read_point_cloud(self.target_pcd_path)
            rospy.loginfo("Loaded target map: %d points from %s",
                         len(self.target_pcd.points), self.target_pcd_path)
        else:
            rospy.logerr("Target map not found at: %s", self.target_pcd_path)
            self.target_pcd = None

        # Publisher for init pose
        self.init_pose_pub = rospy.Publisher(
            'init_pose', PoseStamped, queue_size=1, latch=True
        )

        self.pc_sub = rospy.Subscriber(
            self.input_topic,
            PointCloud2,
            self.point_cloud_callback
        )

        rospy.loginfo("Point cloud accumulation node started.")
        rospy.loginfo("Listening on: %s", self.input_topic)
        rospy.loginfo("Will accumulate %d frames before registration", self.num_frames)

    def point_cloud_callback(self, msg):
        """Callback for receiving and accumulating point cloud messages."""
        if self.registration_done:
            return
        try:
            rospy.loginfo("Received frame %d: %d points",
                         self.frame_count + 1, msg.width * msg.height)

            # Convert ROS point cloud to numpy array
            pcd = point_cloud2_to_open3d(msg)
            points = np.asarray(pcd.points)

            # Accumulate points
            self.accumulated_points.append(points)
            self.frame_count += 1

            rospy.loginfo("Accumulated %d/%d frames", self.frame_count, self.num_frames)

            # Perform registration when enough frames are accumulated
            if self.frame_count >= self.num_frames:
                self.perform_registration()

        except Exception as e:
            rospy.logerr("Error in point cloud callback: %s", str(e))

    def publish_relocalization_msg(self, transformation):
        """
        Publish relocalization message from transformation matrix.

        Args:
            transformation: 4x4 transformation matrix
        """
        try:
            # Extract position from transformation matrix
            x = transformation[0, 3]
            y = transformation[1, 3]
            z = transformation[2, 3]

            # Extract rotation and convert to Euler angles (roll, pitch, yaw)
            roll, pitch, yaw = euler_from_matrix(transformation)

            rospy.loginfo("Publishing init pose: x=%.3f, y=%.3f, z=%.3f, roll=%.3f, pitch=%.3f, yaw=%.3f",
                         x, y, z, np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw))

            # Create PoseStamped message
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

            # Publish
            self.init_pose_pub.publish(init_pose)
            rospy.loginfo("Published init pose to /init_pose")

        except Exception as e:
            rospy.logerr("Error publishing relocalization message: %s", str(e))

    def perform_registration(self):
        """Perform registration with accumulated point cloud."""
        if self.target_pcd is None:
            rospy.logerr("Cannot perform registration: target map not loaded")
            return

        try:
            rospy.loginfo("="*50)
            rospy.loginfo("Accumulated %d frames, performing registration...", self.num_frames)

            # Merge accumulated points
            merged_points = np.vstack(self.accumulated_points)
            rospy.loginfo("Total accumulated points: %d", len(merged_points))

            # Create source point cloud
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(merged_points)

            # Perform registration
            transformation, fitness, rmse = coarse_registration_with_pcd(
                source_pcd, self.target_pcd, self.voxel_size
            )

            rospy.loginfo("="*50)
            rospy.loginfo("Registration completed!")
            rospy.loginfo("Final fitness: %.4f", fitness)
            rospy.loginfo("Final RMSE: %.4f", rmse)
            rospy.loginfo("Final transformation:\n%s", transformation)

            # Publish relocalization message
            self.publish_relocalization_msg(transformation)

            self.registration_done = True
            rospy.loginfo("Registration complete. Unsubscribing from topic.")
            self.pc_sub.unregister()

        except Exception as e:
            rospy.logerr("Error during registration: %s", str(e))

def coarse_registration(pcd_dir, source_file, target_file, voxel_size=0.5,
                       visualize=False, save_result=False, output_file=None):
    """
    Perform coarse registration between two point clouds.

    Args:
        pcd_dir: Directory containing PCD files
        source_file: Source point cloud filename
        target_file: Target point cloud filename
        voxel_size: Voxel size for downsampling
        visualize: Whether to visualize the result
        save_result: Whether to save the transformed source cloud
        output_file: Output filename for saved result

    Returns:
        transformation: 4x4 transformation matrix
        fitness: Registration fitness score
    """
    # Load point clouds
    source, target = load_pcd_files(pcd_dir, source_file, target_file)

    # Preprocess and compute features
    print("\nPreprocessing point clouds...")
    print(f"  Source: no downsampling")
    print(f"  Target: voxel_size={voxel_size:.3f}")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, downsample=False)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, downsample=True)

    # Global registration using RANSAC
    print("\n=== Global Registration (RANSAC) ===")
    result_ransac = execute_global_registration(
        source_down, source_fpfh, target_down, target_fpfh, voxel_size
    )

    print(f"RANSAC Result:")
    print(f"  Fitness: {result_ransac.fitness:.4f}")
    print(f"  RMSE: {result_ransac.inlier_rmse:.4f}")
    print(f"  Transformation:\n{result_ransac.transformation}")

    # ICP refinement
    print("\n=== Local Registration (ICP) ===")
    result_icp = execute_icp_refinement(
        source_down, target_down, result_ransac.transformation, voxel_size
    )

    print(f"ICP Result:")
    print(f"  Fitness: {result_icp.fitness:.4f}")
    print(f"  RMSE: {result_icp.inlier_rmse:.4f}")
    print(f"  Transformation:\n{result_icp.transformation}")

    # Visualize result
    if visualize:
        print("\nVisualizing registration result...")
        visualize_registration(source, target, result_icp.transformation)

    # Save transformed source cloud
    if save_result and output_file:
        source_transformed = source.clone()
        source_transformed.transform(result_icp.transformation)
        output_path = os.path.join(pcd_dir, output_file)
        o3d.io.write_point_cloud(output_path, source_transformed)
        print(f"\nSaved transformed source to: {output_path}")

    return result_icp.transformation, result_icp.fitness, result_icp.inlier_rmse


if __name__ == "__main__":
    # Get PCD directory
    script_dir = get_script_dir()
    pcd_dir = os.path.join(script_dir, "../PCD")
    target_pcd_path = os.path.join(pcd_dir, "aero_maze.pcd")

    # Check if running in ROS environment
    use_ros = rospy.get_param("/use_ros_accumulation", True)
    voxel_size = 0.2
    if use_ros:
        # ROS mode: accumulate point clouds from topic and perform registration
        # input_topic = rospy.get_param("common/lid_topic", "/sensor_cloud")
        num_frames = rospy.get_param("/num_frames", 5)
        # voxel_size = rospy.get_param("/voxel_size", 0.2)

        node = PointCloudAccumulationNode(
            target_pcd_path=target_pcd_path,
            voxel_size=voxel_size,
            num_frames=num_frames,
            input_topic="cloud_registered_body",
            pcd_dir=pcd_dir
        )
        rospy.spin()
    else:
        # Standalone mode: perform registration on existing PCD files
        transformation, fitness, rmse = coarse_registration(
            pcd_dir=pcd_dir,
            source_file="extracted_frames_1.pcd",
            target_file="aero_maze.pcd",
            voxel_size=voxel_size,
            visualize=False,
            save_result=False,
            output_file="registered_source.pcd"
        )

        print("\n" + "="*50)
        print("Registration completed successfully!")
        print(f"Final fitness: {fitness:.4f}")
        print(f"Final RMSE: {rmse:.4f}")
