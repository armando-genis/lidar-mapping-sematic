#include "multicamera_subscriber.hpp"

#include <iomanip>
#include <pcl/common/transforms.h>

namespace multicamera_lidar_calibration
{

namespace
{
void printMatLikePython(std::ostream& out, const cv::Mat& M, const char* name)
{
  out << name << ":\n";
  if (M.rows == 3 && M.cols == 3)
  {
    for (int r = 0; r < 3; r++)
    {
      out << (r == 0 ? "[[" : " [");
      for (int c = 0; c < 3; c++)
        out << std::setw(12) << std::fixed << std::setprecision(8) << M.at<double>(r, c);
      out << (r == 2 ? "]]" : "]\n");
    }
    out << "\n\n";
  }
  else if (M.rows == 3 && M.cols == 1)
  {
    out << "[[ " << std::setw(11) << std::fixed << std::setprecision(8) << M.at<double>(0, 0) << "]\n";
    out << " [" << std::setw(12) << std::fixed << std::setprecision(8) << M.at<double>(1, 0) << "]\n";
    out << " [" << std::setw(12) << std::fixed << std::setprecision(8) << M.at<double>(2, 0) << "]]\n\n";
  }
}

void printMatScientific(std::ostream& out, const cv::Mat& M)
{
  out << std::scientific << std::setprecision(8);
  if (M.rows == 3 && M.cols == 3)
  {
    for (int r = 0; r < 3; r++)
    {
      out << (r == 0 ? "[[" : " [");
      for (int c = 0; c < 3; c++)
        out << std::setw(17) << M.at<double>(r, c);
      out << (r == 2 ? "]]" : "]\n");
    }
    out << "\n\n";
  }
  else if (M.rows == 4 && M.cols == 1)
  {
    for (int r = 0; r < 4; r++)
    {
      out << (r == 0 ? "[[ " : " [") << std::setw(12) << M.at<double>(r, 0);
      out << (r == 3 ? "]]" : " ]") << "\n";
    }
    out << "\n";
  }
  out << std::fixed;
}
}  // namespace

MultiCameraSubscriber::MultiCameraSubscriber() : Node("multicamera_lidar_calibration_node")
{
  this->declare_parameter<int>("num_cameras", 3);
  this->declare_parameter<std::string>("calib_dir", "");
  this->declare_parameter<std::string>("lidar_topic", "");
  this->declare_parameter<double>("lidar_rotation_x", 0.0);
  this->declare_parameter<double>("lidar_rotation_y", 0.0);
  this->declare_parameter<double>("lidar_rotation_z", 0.0);

  num_cameras_ = this->get_parameter("num_cameras").as_int();
  calib_dir_ = this->get_parameter("calib_dir").as_string();
  lidar_topic_ = this->get_parameter("lidar_topic").as_string();
  lidar_rotation_x_ = static_cast<float>(this->get_parameter("lidar_rotation_x").as_double());
  lidar_rotation_y_ = static_cast<float>(this->get_parameter("lidar_rotation_y").as_double());
  lidar_rotation_z_ = static_cast<float>(this->get_parameter("lidar_rotation_z").as_double());

  // Build rotation matrix: Z * Y * X (same order as semantic_lidar_slam)
  Eigen::Matrix3f R =
    (Eigen::AngleAxisf(lidar_rotation_z_, Eigen::Vector3f::UnitZ()) *
     Eigen::AngleAxisf(lidar_rotation_y_, Eigen::Vector3f::UnitY()) *
     Eigen::AngleAxisf(lidar_rotation_x_, Eigen::Vector3f::UnitX()))
    .toRotationMatrix();
  rotation_matrix_ = Eigen::Matrix4f::Identity();
  rotation_matrix_.block<3, 3>(0, 0) = R;

  std::cout << "lidar_topic:" << lidar_topic_ << std::endl;
  std::cout << "lidar_rotation: x=" << lidar_rotation_x_
            << " y=" << lidar_rotation_y_
            << " z=" << lidar_rotation_z_ << std::endl;


  calib.load(calib_dir_);

  if (static_cast<size_t>(0) < calib.camera_array.size() && calib.camera_array[0])
  {
    auto* cam = calib.camera_array[0].get();
    std::cout << "\nIntrinsics: LOADED\n\n";
    cv::Size sz = cam->get_frame_size();
    std::cout << "Frame size (w x h): (" << sz.width << ", " << sz.height << ")\n\n";
    std::cout << "Intrinsics (K):\n";
    printMatScientific(std::cout, cam->get_K());
    std::cout << "Distortion (D):\n";
    printMatScientific(std::cout, cam->get_D());
  }

  if (static_cast<size_t>(0) < calib.extrinsics_array.size() && calib.extrinsics_array[0])
  {
    auto* ext = calib.extrinsics_array[0].get();
    std::cout << "Extrinsics (Lidar \u2192 Camera): LOADED\n\n";
    printMatLikePython(std::cout, ext->get_R_opencv(), "R (opencv_frame)");
    printMatLikePython(std::cout, ext->get_t_opencv(), "t (opencv_frame)");
    printMatLikePython(std::cout, ext->get_R_robot(), "R (robot_frame)");
    printMatLikePython(std::cout, ext->get_t_robot(), "t (robot_frame)");
    const auto& lr = ext->get_lidar_rotation();
    std::cout << "Lidar rotation: {'axis_x': " << std::fixed << std::setprecision(8) << lr.axis_x
              << ", 'axis_y': " << lr.axis_y << ", 'axis_z': " << lr.axis_z << "}\n" << std::endl;
  }

  // lidar_buffer_ is a deque — no pre-init needed

  auto lidar_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  auto camera_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

  // Subscribe to lidar topic
  lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_,
      lidar_qos,
      std::bind(&MultiCameraSubscriber::lidarCallback, this, std::placeholders::_1)
  );


  // Subscribe to camera topics
  for (int i = 0; i < num_cameras_; i++)
  {
    std::string topic = "/racecar/camera/camera_" + std::to_string(i) + "/image_raw";
    const int camera_id = i;

    auto sub = this->create_subscription<sensor_msgs::msg::Image>(
        topic,
        camera_qos,
        [this, camera_id](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          imageCallback(msg, camera_id);
        }
    );

    subscribers_.push_back(sub);

    RCLCPP_INFO(this->get_logger(), "Subscribed to %s", topic.c_str());
  }

  cv::namedWindow("MultiCamera", cv::WINDOW_NORMAL);
}

void MultiCameraSubscriber::imageCallback(
    sensor_msgs::msg::Image::ConstSharedPtr msg, int camera_id)
{
  try
  {
    cv::Mat frame;

    if (msg->encoding == "mjpeg")
    {
      std::vector<uint8_t> buffer(msg->data.begin(), msg->data.end());
      frame = cv::imdecode(buffer, cv::IMREAD_COLOR);
    }
    else if (msg->encoding == "rgb8")
    {
      frame = cv::Mat(msg->height, msg->width, CV_8UC3,
                      const_cast<uint8_t*>(msg->data.data())).clone();
      cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
    }
    else if (msg->encoding == "bgr8")
    {
      frame = cv::Mat(msg->height, msg->width, CV_8UC3,
                      const_cast<uint8_t*>(msg->data.data())).clone();
    }
    else if (msg->encoding == "mono8")
    {
      frame = cv::Mat(msg->height, msg->width, CV_8UC1,
                      const_cast<uint8_t*>(msg->data.data())).clone();
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "Unsupported encoding: %s", msg->encoding.c_str());
      return;
    }

    if (frame.empty())
      return;

    // 1) Undistort first (required: projection uses pinhole K on undistorted image)
    if (static_cast<size_t>(camera_id) < calib.camera_array.size())
    {
      auto cam = calib.camera_array[camera_id];

      if (cam)
      {
        cam->ensure_size(frame.cols, frame.rows);
        frame = cam->undistort(frame);
      }
    }

    // 2) Find the closest lidar cloud to this image's timestamp
    const double image_stamp = static_cast<double>(msg->header.stamp.sec) +
                               static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    RCLCPP_INFO(this->get_logger(), "[CAM %d] image received, stamp=%.3f, encoding='%s'",
                camera_id, image_stamp, msg->encoding.c_str());

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_copy;
    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      RCLCPP_INFO(this->get_logger(), "[CAM %d] lidar_buffer size=%zu", camera_id, lidar_buffer_.size());

      double best_dt = std::numeric_limits<double>::max();
      const LidarFrame* best = nullptr;
      for (const auto& lf : lidar_buffer_)
      {
        double dt = std::abs(lf.stamp - image_stamp);
        if (dt < best_dt)
        {
          best_dt = dt;
          best = &lf;
        }
      }
      // Accept within 500ms — camera and lidar clocks have a ~350ms fixed offset in the bag
      if (best_dt < 0.5 && best != nullptr)
      {
        cloud_copy = best->cloud;
        RCLCPP_INFO(this->get_logger(), "[CAM %d] matched lidar cloud: %zu pts, dt=%.3fs",
                    camera_id, cloud_copy->size(), best_dt);
      }
      else
      {
        RCLCPP_WARN(this->get_logger(), "[CAM %d] NO lidar match (best_dt=%.3fs > 0.5s, buffer=%zu)",
                    camera_id, best_dt, lidar_buffer_.size());
      }
    }

    if (static_cast<size_t>(camera_id) < calib.extrinsics_array.size())
    {
      auto ext = calib.extrinsics_array[camera_id];

      if (ext && cloud_copy && !cloud_copy->empty())
      {
        cv::Mat R = ext->get_R_opencv();
        cv::Mat t = ext->get_t_opencv();

        cv::Mat K = calib.camera_array[camera_id]->get_K();

        // 3) Project lidar onto undistorted frame (same order as Python)
        frame = projectLidarOnImage(frame, cloud_copy, R, t, K);
      }
    }

    cv::putText(
        frame,
        "Camera " + std::to_string(camera_id),
        {20, 40},
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        {0, 255, 0},
        2
    );

    cv::imshow("Camera " + std::to_string(camera_id), frame);
    cv::waitKey(1);
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(this->get_logger(), "Error decoding image: %s", e.what());
  }
}

void MultiCameraSubscriber::lidarCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "[LIDAR] msg received: %zu bytes, frame_id='%s'",
              msg->data.size(), msg->header.frame_id.c_str());

  if (msg->data.empty())
  {
    RCLCPP_WARN(this->get_logger(), "[LIDAR] msg data is empty, skipping");
    return;
  }

  LidarFrame lf;
  lf.stamp = static_cast<double>(msg->header.stamp.sec) +
             static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
  lf.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::fromROSMsg(*msg, *lf.cloud);
  pcl::transformPointCloud(*lf.cloud, *lf.cloud, rotation_matrix_);

  RCLCPP_INFO(this->get_logger(), "[LIDAR] decoded %zu points, stamp=%.3f",
              lf.cloud->size(), lf.stamp);

  std::lock_guard<std::mutex> lock(lidar_mutex_);
  lidar_buffer_.push_back(std::move(lf));
  while (lidar_buffer_.size() > 15)
    lidar_buffer_.pop_front();
  RCLCPP_INFO(this->get_logger(), "[LIDAR] buffer size: %zu", lidar_buffer_.size());
}

cv::Mat MultiCameraSubscriber::projectLidarOnImage(
    const cv::Mat& img,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Mat& K,
    bool debug_print)
{
    cv::Mat out = img.clone();

    if (!cloud || cloud->points.empty())
        return out;

    const double fx = K.at<double>(0, 0);
    const double fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2);
    const double cy = K.at<double>(1, 2);

    int h = out.rows;
    int w = out.cols;

    for (const auto& p : cloud->points)
    {
        cv::Mat pt = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);

        cv::Mat cam = R * pt + t;

        double x = cam.at<double>(0);
        double y = cam.at<double>(1);
        double z = cam.at<double>(2);

        // Skip points that are too close
        if (z <= 0.1)
            continue;

        // Project the 3D point into 2D image coordinates
        int u = static_cast<int>(fx * x / z + cx + 0.5);
        int v = static_cast<int>(fy * y / z + cy + 0.5);

        // Check if the projected point is inside the image bounds
        if (u < 0 || u >= w || v < 0 || v >= h)
            continue;

        // Normalize depth for visualization
        float depth = std::min(static_cast<float>(z) / 20.0f, 1.0f);

        // Color the point based on its depth (close points are red, far points are blue)
        cv::Scalar color(
            255,                                  // Red channel
            static_cast<int>(255 * (1.0 - depth)), // Green channel (inverse depth)
            static_cast<int>(255 * depth)          // Blue channel (depth)
        );

        // Draw the point on the image
        cv::circle(out, {u, v}, 2, color, -1);
    }

    return out;
}

}  // namespace multicamera_lidar_calibration

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<multicamera_lidar_calibration::MultiCameraSubscriber>());
  rclcpp::shutdown();
  return 0;
}
