#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <cmath>
#include <limits>

namespace py = pybind11;

// Convert numpy array (N x 8) to PCL PointXYZ cloud (using only x, y, z)
pcl::PointCloud<pcl::PointXYZ>::Ptr numpy_to_pcl_xyz(py::array_t<float> input) {
    py::buffer_info buf = input.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }

    if (buf.shape[1] != 8) {
        throw std::runtime_error("Input array must have 8 columns (x, y, z, intensity, normal_x, normal_y, normal_z, curvature)");
    }

    if (buf.shape[0] == 0) {
        throw std::runtime_error("Input array is empty");
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    float* ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < buf.shape[0]; ++i) {
        float x = ptr[i * 8 + 0];
        float y = ptr[i * 8 + 1];
        float z = ptr[i * 8 + 2];

        // Only add finite points
        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            cloud->points.push_back(pcl::PointXYZ(x, y, z));
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

// Downsample point cloud (PointXYZ version)
pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_xyz(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    float leaf_size) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*filtered);
    return filtered;
}

// Convert 6DOF pose (x, y, z, roll, pitch, yaw) to 4x4 transformation matrix
Eigen::Matrix4f pose_to_matrix(float x, float y, float z, float roll, float pitch, float yaw) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

    // Translation
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;

    // Rotation using Euler angles (ZYX convention)
    Eigen::AngleAxisf roll_angle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch_angle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yaw_angle(yaw, Eigen::Vector3f::UnitZ());

    Eigen::Quaternionf q = yaw_angle * pitch_angle * roll_angle;
    T.block<3, 3>(0, 0) = q.matrix();

    return T;
}

// Batch GICP matching: match multiple point clouds to the first one
// init_poses: N x 6 array (x, y, z, roll, pitch, yaw) for initial guesses
py::list batch_match(py::list clouds_np, py::array_t<float> init_poses,
                         float max_correspondence_distance = 1.0,
                         float trans_epsilon = 1e-6, int max_iterations = 1000,
                         float downsample_leaf_size = 0.5) {

    size_t num_clouds = clouds_np.size();
    if (num_clouds == 0) {
        throw std::runtime_error("No point clouds provided");
    }

    // Parse initial poses
    py::buffer_info pose_buf = init_poses.request();
    std::vector<Eigen::Matrix4f> initial_guesses;

    size_t num_poses = pose_buf.shape[0];
    float* pose_ptr = static_cast<float*>(pose_buf.ptr);

    for (size_t i = 0; i < num_poses; ++i) {
        float x = pose_ptr[i * 6 + 0];
        float y = pose_ptr[i * 6 + 1];
        float z = pose_ptr[i * 6 + 2];
        float roll = pose_ptr[i * 6 + 3];
        float pitch = pose_ptr[i * 6 + 4];
        float yaw = pose_ptr[i * 6 + 5];

        Eigen::Matrix4f T = pose_to_matrix(x, y, z, roll, pitch, yaw);
        if (i) initial_guesses.push_back(initial_guesses[0].inverse() * T);
        else initial_guesses.push_back(T);
    }

    // First cloud is the target
    py::array_t<float> target_np = clouds_np[0].cast<py::array_t<float>>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud = numpy_to_pcl_xyz(target_np);

    if (downsample_leaf_size > 0.0f)
        target_cloud = downsample_xyz(target_cloud, downsample_leaf_size);

    if (target_cloud->points.size() < 100) {
        throw std::runtime_error("Target cloud has too few points after downsampling.");
    }

    py::list results;

    // First cloud has identity transform
    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    py::list first_result;
    first_result.append(identity);
    first_result.append(true);
    first_result.append(0.0f);
    results.append(first_result);

    // Match each subsequent cloud to the first using GICP
    for (size_t i = 1; i < num_clouds; ++i) {
        py::array_t<float> source_np = clouds_np[i].cast<py::array_t<float>>();
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = numpy_to_pcl_xyz(source_np);

        if (downsample_leaf_size > 0.0f)
            source_cloud = downsample_xyz(source_cloud, downsample_leaf_size);

        if (source_cloud->points.size() < 100) {
            Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
            py::list result;
            result.append(identity);
            result.append(false);
            result.append(std::numeric_limits<float>::max());
            results.append(result);
            continue;
        }

        // Initialize GICP
        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;

        // Set GICP parameters
        gicp.setMaxCorrespondenceDistance(max_correspondence_distance);
        gicp.setTransformationEpsilon(trans_epsilon);
        gicp.setMaximumIterations(max_iterations);

        // Set input clouds
        gicp.setInputSource(source_cloud);
        gicp.setInputTarget(target_cloud);

        // Get initial guess (use provided pose or identity)
        Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();
        if (i < initial_guesses.size()) {
            init_guess = initial_guesses[i];
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);

        gicp.align(*output, init_guess);

        // Get final transformation
        Eigen::Matrix4f transformation = gicp.getFinalTransformation();
        bool converged = gicp.hasConverged();
        float fitness_score = gicp.getFitnessScore();

        py::list result;
        result.append(transformation);
        result.append(converged);
        result.append(fitness_score);
        results.append(result);
    }

    return results;
}

PYBIND11_MODULE(pycloud_matcher, m) {
    m.doc() = "GICP point cloud matching module";

    m.def("batch_match", &batch_match,
          py::arg("clouds"),
          py::arg("init_poses"),
          py::arg("max_correspondence_distance") = 1.0,
          py::arg("trans_epsilon") = 1e-6,
          py::arg("max_iterations") = 1000,
          py::arg("downsample_leaf_size") = 0.5,
          "Perform batch GICP matching: match all clouds to the first cloud");
}
