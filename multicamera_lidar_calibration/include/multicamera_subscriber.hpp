#ifndef MULTICAMERA_LIDAR_CALIBRATION__MULTICAMERA_SUBSCRIBER_HPP_
#define MULTICAMERA_LIDAR_CALIBRATION__MULTICAMERA_SUBSCRIBER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <mutex>
#include <memory>
#include <atomic>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include "CameraLidarExtrinsics.hpp"
#include "CameraUndistorter.hpp"
#include "DataLoader.hpp"
#include "edge_calibrator.hpp"

namespace multicamera_lidar_calibration
{

class MultiCameraSubscriber : public rclcpp::Node
{
public:
  explicit MultiCameraSubscriber();

private:
  void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg, int camera_id);
  void lidarCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  cv::Mat projectLidarOnImage(
      const cv::Mat& img,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
      const cv::Mat& R,
      const cv::Mat& t,
      const cv::Mat& K,
      bool debug_print = false);

  void tryCalibrate(int camera_id,
                    const cv::Mat& undistorted_frame,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

  // Save full-res image + edge image + PCD to disk
  void saveDebugFrame(int camera_id,
                      int frame_idx,
                      const cv::Mat& undistorted,
                      const cv::Mat& edge_image,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

  // ── Parameters ─────────────────────────────────────────────────────────────
  int         num_cameras_;
  std::string calib_dir_;
  std::string lidar_topic_;
  bool        calibration_mode_;
  double      display_scale_;
  bool        save_debug_frames_;
  std::string debug_output_dir_;
  double      sync_max_dt_;
  int         sync_wait_frames_;

  // ── Subscribers ────────────────────────────────────────────────────────────
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subscribers_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr        lidar_sub_;

  // ── LiDAR buffer ───────────────────────────────────────────────────────────
  struct LidarFrame {
    double stamp;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  };
  std::deque<LidarFrame> lidar_buffer_;
  std::mutex             lidar_mutex_;

  Eigen::Matrix4f rotation_matrix_;
  float lidar_rotation_x_{0.0f};
  float lidar_rotation_y_{0.0f};
  float lidar_rotation_z_{0.0f};

  CalibrationLoader           calib;
  std::vector<EdgeCalibrator> calibrators_;

  // Per-camera frame counter; atomics in unique_ptr — std::vector<std::atomic<>> is not usable
  std::vector<std::unique_ptr<std::atomic<int>>> frame_counters_;
};

}  // namespace multicamera_lidar_calibration

#endif  // MULTICAMERA_LIDAR_CALIBRATION__MULTICAMERA_SUBSCRIBER_HPP_