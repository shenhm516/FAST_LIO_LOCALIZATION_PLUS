#!/usr/bin/env python3
"""
Extract first N point cloud frames from a rosbag and save as PCD file.
"""
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import argparse
import os
from scipy.spatial.transform import Rotation


def get_script_dir():
    """Get the directory where this script is located."""
    return os.path.dirname(os.path.abspath(__file__))


def point_cloud2_to_array(cloud_msg):
    """Convert sensor_msgs/PointCloud2 to numpy array."""
    # Read the point cloud data
    gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(gen))

    return points


def get_extrinsic():
    """Get extrinsic transformation matrix (4x4) from Euler angles and translation.

    Edit the roll, pitch, yaw (in degrees) and translation values below
    to match your sensor configuration.
    """
    # ========== USER CONFIGURATION ==========
    # Euler angles in degrees (ZYX/RPY convention: roll, pitch, yaw)
    roll = 0.0   # rotation around X axis
    pitch = 30.0  # rotation around Y axis
    yaw = 0.0    # rotation around Z axis

    # Translation (x, y, z)
    tx, ty, tz = 0.0, 0.0, 0.0
    # =======================================

    # Create rotation from Euler angles (ZYX sequence, degrees)
    rotation = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    R = rotation.as_matrix()

    # Build 4x4 transformation matrix: [[R, t], [0, 1]]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = [tx, ty, tz]

    return extrinsic


def transform_points(points, transform):
    """Transform points using a 4x4 transformation matrix.

    Args:
        points: Nx3 numpy array of points
        transform: 4x4 transformation matrix

    Returns:
        Transformed Nx3 numpy array
    """
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    # Apply transformation: points_new = T * points
    transformed = (transform @ points_homo.T).T
    return transformed[:, :3]


def extract_first_frames(bag_path, output_pcd_path, num_frames=5, topic="/velodyne_points"):
    """
    Extract first N point cloud frames from a rosbag and merge into one PCD file.

    Args:
        bag_path: Path to the rosbag file
        output_pcd_path: Path to save the output PCD file
        num_frames: Number of frames to extract (default: 5)
        topic: Point cloud topic name (default: /velodyne_points)
    """
    print(f"Opening rosbag: {bag_path}")
    bag = rosbag.Bag(bag_path)

    # Get available topics
    print("Available topics in bag:")
    info = bag.get_type_and_topic_info()
    for topic_name, topic_info in info.topics.items():
        if 'PointCloud' in topic_info.msg_type:
            print(f"  - {topic_name} ({topic_info.msg_type})")

    # Extract point clouds
    point_clouds = []
    frame_count = 0

    print(f"\nExtracting first {num_frames} frames from topic: {topic}")

    for topic, msg, t in bag.read_messages(topics=[topic]):
        if frame_count >= num_frames:
            break

        try:
            points = point_cloud2_to_array(msg)
            if len(points) > 0:
                print(f"  Frame {frame_count + 1}: {len(points)} points")
                point_clouds.append(points)
                frame_count += 1
        except Exception as e:
            print(f"  Warning: Could not process frame: {e}")
            continue

    bag.close()

    if frame_count == 0:
        print("Error: No point cloud frames found!")
        return False

    # Merge all point clouds
    print(f"\nMerging {frame_count} frames...")
    merged_points = np.vstack(point_clouds)
    print(f"Total points: {len(merged_points)}")

    # Apply extrinsic transformation
    extrinsic = get_extrinsic()
    print("Applying extrinsic transformation...")
    transformed_points = transform_points(merged_points, extrinsic)

    # Create Open3D point cloud and save
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_points)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_pcd_path), exist_ok=True)

    # Save PCD file
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"Saved PCD file to: {output_pcd_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract first N point cloud frames from a rosbag and save as PCD."
    )
    parser.add_argument("--bag_path", type=str, default="/mnt/e/aeromaze/2026-03-11-13-40-24.bag")
    parser.add_argument("--output", type=str, default="/mnt/Y/rosbag/UniLPR/AEROMAZE/extracted_frames_0.pcd",
                        help="Output PCD file path (default: PCD/extracted_frames.pcd)")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of frames to extract (default: 5)")
    parser.add_argument("--topic", type=str, default="/quad0_pcl_render_node/sensor_cloud",
                        help="Point cloud topic name (default: /velodyne_points)")

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        script_dir = get_script_dir()
        args.output = os.path.join(script_dir, "../PCD/extracted_frames.pcd")

    # Extract and save
    extract_first_frames(args.bag_path, args.output, args.num_frames, args.topic)
