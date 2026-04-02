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
    out << " ["  << std::setw(12) << std::fixed << std::setprecision(8) << M.at<double>(1, 0) << "]\n";
    out << " ["  << std::setw(12) << std::fixed << std::setprecision(8) << M.at<double>(2, 0) << "]]\n\n";
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

// ── Bottom panel: white bg + black Canny edges + coloured LiDAR dots ─────────
cv::Mat buildEdgeDebugFrame(
    const cv::Mat& undistorted,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Mat& K)
{
  cv::Mat gray, blurred, edges_raw;
  if (undistorted.channels() == 3)
    cv::cvtColor(undistorted, gray, cv::COLOR_BGR2GRAY);
  else
    gray = undistorted.clone();
  cv::GaussianBlur(gray, blurred, {5, 5}, 1.5);
  cv::Canny(blurred, edges_raw, 50, 150);

  // White background, black edges
  cv::Mat debug(undistorted.rows, undistorted.cols, CV_8UC3, cv::Scalar(255, 255, 255));
  for (int r = 0; r < edges_raw.rows; ++r)
    for (int c = 0; c < edges_raw.cols; ++c)
      if (edges_raw.at<uint8_t>(r, c) > 0)
        debug.at<cv::Vec3b>(r, c) = {0, 0, 0};

  if (!cloud || cloud->empty() || R.empty() || t.empty() || K.empty())
    return debug;

  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);
  const int w = debug.cols;
  const int h = debug.rows;

  for (const auto& p : cloud->points)
  {
    cv::Mat pt  = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
    cv::Mat cam = R * pt + t;
    double z = cam.at<double>(2);
    if (z < 0.1) continue;
    int u = static_cast<int>(fx * cam.at<double>(0) / z + cx + 0.5);
    int v = static_cast<int>(fy * cam.at<double>(1) / z + cy + 0.5);
    if (u < 0 || u >= w || v < 0 || v >= h) continue;

    float depth = std::min(static_cast<float>(z) / 20.0f, 1.0f);
    cv::Scalar color(
        static_cast<int>(255 * depth),
        static_cast<int>(255 * (1.0f - depth)),
        255);
    cv::circle(debug, {u, v}, 3, color, -1);
  }
  return debug;
}

// ── Scale a mat down if display_scale < 1.0 ──────────────────────────────────
cv::Mat scaleForDisplay(const cv::Mat& src, double scale)
{
  if (std::abs(scale - 1.0) < 1e-3 || scale <= 0.0)
    return src;
  cv::Mat out;
  cv::resize(src, out, {}, scale, scale, cv::INTER_AREA);
  return out;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
MultiCameraSubscriber::MultiCameraSubscriber()
: Node("multicamera_lidar_calibration_node")
{
  this->declare_parameter<int>("num_cameras", 3);
  this->declare_parameter<std::string>("calib_dir", "");
  this->declare_parameter<std::string>("lidar_topic", "");
  this->declare_parameter<double>("lidar_rotation_x", 0.0);
  this->declare_parameter<double>("lidar_rotation_y", 0.0);
  this->declare_parameter<double>("lidar_rotation_z", 0.0);
  this->declare_parameter<bool>("calibration_mode", false);
  this->declare_parameter<double>("display_scale", 0.4);   // ← NEW

  num_cameras_      = this->get_parameter("num_cameras").as_int();
  calib_dir_        = this->get_parameter("calib_dir").as_string();
  lidar_topic_      = this->get_parameter("lidar_topic").as_string();
  calibration_mode_ = this->get_parameter("calibration_mode").as_bool();
  display_scale_    = this->get_parameter("display_scale").as_double();  // ← NEW
  lidar_rotation_x_ = static_cast<float>(this->get_parameter("lidar_rotation_x").as_double());
  lidar_rotation_y_ = static_cast<float>(this->get_parameter("lidar_rotation_y").as_double());
  lidar_rotation_z_ = static_cast<float>(this->get_parameter("lidar_rotation_z").as_double());

  Eigen::Matrix3f R =
    (Eigen::AngleAxisf(lidar_rotation_z_, Eigen::Vector3f::UnitZ()) *
     Eigen::AngleAxisf(lidar_rotation_y_, Eigen::Vector3f::UnitY()) *
     Eigen::AngleAxisf(lidar_rotation_x_, Eigen::Vector3f::UnitX()))
    .toRotationMatrix();
  rotation_matrix_ = Eigen::Matrix4f::Identity();
  rotation_matrix_.block<3, 3>(0, 0) = R;

  std::cout << "\n================================================\n";
  std::cout << "  multicamera_lidar_calibration starting\n";
  std::cout << "  lidar_topic:      " << lidar_topic_      << "\n";
  std::cout << "  calib_dir:        " << calib_dir_        << "\n";
  std::cout << "  calibration_mode: " << (calibration_mode_ ? "ON" : "OFF") << "\n";
  std::cout << "  display_scale:    " << display_scale_    << "\n";
  std::cout << "  lidar_rotation:   x=" << lidar_rotation_x_
            << " y=" << lidar_rotation_y_
            << " z=" << lidar_rotation_z_ << "\n";
  std::cout << "================================================\n\n";

  calib.load(calib_dir_);

  if (!calib.camera_array.empty() && calib.camera_array[0])
  {
    auto* cam = calib.camera_array[0].get();
    cv::Size sz = cam->get_frame_size();
    std::cout << "Intrinsics cam0: " << sz.width << "x" << sz.height << "\n";
    printMatScientific(std::cout, cam->get_K());
  }
  if (!calib.extrinsics_array.empty() && calib.extrinsics_array[0])
  {
    auto* ext = calib.extrinsics_array[0].get();
    std::cout << "Extrinsics cam0:\n";
    printMatLikePython(std::cout, ext->get_R_opencv(), "R");
    printMatLikePython(std::cout, ext->get_t_opencv(), "t");
  }

  EdgeCalibrator::Config cal_cfg;
  cal_cfg.min_frames        = 10;
  cal_cfg.max_frames        = 50;
  cal_cfg.canny_low         = 50;
  cal_cfg.canny_high        = 150;
  cal_cfg.depth_edge_thresh = 0.5;
  cal_cfg.lidar_sample_step = 3;
  for (int i = 0; i < num_cameras_; ++i)
    calibrators_.emplace_back(cal_cfg);

  auto lidar_qos  = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  auto camera_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();

  lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, lidar_qos,
      std::bind(&MultiCameraSubscriber::lidarCallback, this, std::placeholders::_1));

  for (int i = 0; i < num_cameras_; i++)
  {
    std::string topic = "/racecar/camera/camera_" + std::to_string(i) + "/image_raw";
    const int camera_id = i;
    auto sub = this->create_subscription<sensor_msgs::msg::Image>(
        topic, camera_qos,
        [this, camera_id](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          imageCallback(msg, camera_id);
        });
    subscribers_.push_back(sub);
    RCLCPP_INFO(this->get_logger(), "Subscribed to %s", topic.c_str());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// imageCallback
// ─────────────────────────────────────────────────────────────────────────────
void MultiCameraSubscriber::imageCallback(
    sensor_msgs::msg::Image::ConstSharedPtr msg, int camera_id)
{
  try
  {
    // ── Decode ────────────────────────────────────────────────────────────────
    cv::Mat frame;
    if (msg->encoding == "mjpeg")
    {
      std::vector<uint8_t> buf(msg->data.begin(), msg->data.end());
      frame = cv::imdecode(buf, cv::IMREAD_COLOR);
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
    if (frame.empty()) return;

    // ── 1) Undistort ──────────────────────────────────────────────────────────
    if (static_cast<size_t>(camera_id) < calib.camera_array.size() &&
        calib.camera_array[camera_id])
    {
      auto cam = calib.camera_array[camera_id];
      cam->ensure_size(frame.cols, frame.rows);
      frame = cam->undistort(frame);
    }

    // ── 2) Match LiDAR cloud ──────────────────────────────────────────────────
    const double image_stamp = static_cast<double>(msg->header.stamp.sec) +
                               static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_copy;
    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      double best_dt = std::numeric_limits<double>::max();
      const LidarFrame* best = nullptr;
      for (const auto& lf : lidar_buffer_)
      {
        double dt = std::abs(lf.stamp - image_stamp);
        if (dt < best_dt) { best_dt = dt; best = &lf; }
      }
      if (best_dt < 0.5 && best != nullptr)
        cloud_copy = best->cloud;
      else
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[CAM %d] No lidar match (dt=%.3fs, buf=%zu)",
            camera_id, best_dt, lidar_buffer_.size());
    }

    // ── 3) Get R, t, K ────────────────────────────────────────────────────────
    cv::Mat R, t, K;
    if (static_cast<size_t>(camera_id) < calib.extrinsics_array.size() &&
        calib.extrinsics_array[camera_id])
    {
      R = calib.extrinsics_array[camera_id]->get_R_opencv();
      t = calib.extrinsics_array[camera_id]->get_t_opencv();
    }
    if (static_cast<size_t>(camera_id) < calib.camera_array.size() &&
        calib.camera_array[camera_id])
      K = calib.camera_array[camera_id]->get_K();

    // ── 4) Calibration ────────────────────────────────────────────────────────
    if (calibration_mode_ && cloud_copy && !cloud_copy->empty() &&
        !R.empty() && !t.empty() && !K.empty())
    {
      tryCalibrate(camera_id, frame, cloud_copy);
      if (calib.extrinsics_array[camera_id])
      {
        R = calib.extrinsics_array[camera_id]->get_R_opencv();
        t = calib.extrinsics_array[camera_id]->get_t_opencv();
      }
    }

    // ── 5) TOP panel: colour image + LiDAR projection ─────────────────────────
    cv::Mat top = frame.clone();
    if (top.channels() == 1)
      cv::cvtColor(top, top, cv::COLOR_GRAY2BGR);
    if (cloud_copy && !cloud_copy->empty() && !R.empty())
      top = projectLidarOnImage(top, cloud_copy, R, t, K);

    // Label — scale font size with display_scale so it stays readable
    {
      double fs = std::max(0.5, 0.9 / display_scale_);  // bigger font on smaller image
      std::string label = "CAM " + std::to_string(camera_id) + " | projected";
      cv::putText(top, label, {10, 35}, cv::FONT_HERSHEY_SIMPLEX, fs, {0, 255, 0}, 2);
    }

    // ── 6) BOTTOM panel: edges + LiDAR dots ──────────────────────────────────
    cv::Mat bottom = buildEdgeDebugFrame(frame, cloud_copy, R, t, K);

    {
      double fs = std::max(0.5, 0.9 / display_scale_);
      if (calibration_mode_ && static_cast<size_t>(camera_id) < calibrators_.size())
      {
        int n    = calibrators_[camera_id].frameCount();
        int need = 10;
        std::string status = "CALIB " + std::to_string(n) + "/" + std::to_string(need);
        cv::Scalar col = (n >= need) ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 140, 255);
        cv::putText(bottom, status, {10, 35}, cv::FONT_HERSHEY_SIMPLEX, fs, col, 2);
      }
      else
      {
        cv::putText(bottom, "Edges + LiDAR",
                    {10, 35}, cv::FONT_HERSHEY_SIMPLEX, fs, {80, 80, 80}, 2);
      }
    }

    // ── 7) Make both panels same type, add separator, stack, scale, show ──────
    if (top.type()    != CV_8UC3) cv::cvtColor(top,    top,    cv::COLOR_GRAY2BGR);
    if (bottom.type() != CV_8UC3) cv::cvtColor(bottom, bottom, cv::COLOR_GRAY2BGR);
    if (top.size()    != bottom.size()) cv::resize(bottom, bottom, top.size());

    // Yellow separator line at the bottom of the top panel
    cv::line(top,
             {0,            top.rows - 2},
             {top.cols - 1, top.rows - 2},
             {0, 220, 220}, 3);

    cv::Mat stacked;
    cv::vconcat(top, bottom, stacked);

    // ── Scale for display ─────────────────────────────────────────────────────
    // At 0.4 scale: 1920x1080 per panel → 768x432 per panel → 768x864 total
    cv::Mat display = scaleForDisplay(stacked, display_scale_);

    std::string win = "Camera " + std::to_string(camera_id);
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);  // AUTOSIZE respects our resize
    cv::imshow(win, display);
    cv::waitKey(1);
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(this->get_logger(), "[CAM %d] Exception: %s", camera_id, e.what());
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// tryCalibrate
// ─────────────────────────────────────────────────────────────────────────────
void MultiCameraSubscriber::tryCalibrate(
    int camera_id,
    const cv::Mat& undistorted_frame,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
  if (static_cast<size_t>(camera_id) >= calibrators_.size())           return;
  if (static_cast<size_t>(camera_id) >= calib.extrinsics_array.size()) return;
  if (static_cast<size_t>(camera_id) >= calib.camera_array.size())     return;

  auto& cal = calibrators_[camera_id];
  auto  ext = calib.extrinsics_array[camera_id];
  cv::Mat K = calib.camera_array[camera_id]->get_K();
  cv::Mat R = ext->get_R_opencv();
  cv::Mat t = ext->get_t_opencv();

  bool added = cal.addFrame(undistorted_frame, cloud, K, R, t);
  if (added)
    std::cout << "[CAM " << camera_id << "] Calib frame "
              << cal.frameCount() << "/10\n";

  if (!cal.hasEnoughFrames()) return;

  std::cout << "[CAM " << camera_id << "] ===== Running Ceres edge optimisation =====\n";
  cv::Mat R_refined, t_refined;

  if (cal.solve(R, t, R_refined, t_refined, true))
  {
    ext->set_R_opencv(R_refined);
    ext->set_t_opencv(t_refined);
    calib.save(calib_dir_);
    std::cout << "[CAM " << camera_id << "] Saved to " << calib_dir_ << "\n";
    RCLCPP_INFO(this->get_logger(), "[CAM %d] Calibration refined and saved!", camera_id);
  }
  else
  {
    std::cout << "[CAM " << camera_id << "] Solve FAILED\n";
    RCLCPP_WARN(this->get_logger(), "[CAM %d] Calibration solve failed.", camera_id);
  }

  cal.reset();
  std::cout << "[CAM " << camera_id << "] Calibrator reset — collecting next batch\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// lidarCallback
// ─────────────────────────────────────────────────────────────────────────────
void MultiCameraSubscriber::lidarCallback(
    sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  if (msg->data.empty())
  {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "[LIDAR] empty, skipping");
    return;
  }

  LidarFrame lf;
  lf.stamp = static_cast<double>(msg->header.stamp.sec) +
             static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
  lf.cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::fromROSMsg(*msg, *lf.cloud);
  pcl::transformPointCloud(*lf.cloud, *lf.cloud, rotation_matrix_);

  std::lock_guard<std::mutex> lock(lidar_mutex_);
  lidar_buffer_.push_back(std::move(lf));
  while (lidar_buffer_.size() > 15)
    lidar_buffer_.pop_front();
}

// ─────────────────────────────────────────────────────────────────────────────
// projectLidarOnImage
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat MultiCameraSubscriber::projectLidarOnImage(
    const cv::Mat& img,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const cv::Mat& R,
    const cv::Mat& t,
    const cv::Mat& K,
    bool /*debug_print*/)
{
  cv::Mat out = img.clone();
  if (!cloud || cloud->points.empty()) return out;

  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);
  const int h = out.rows;
  const int w = out.cols;

  for (const auto& p : cloud->points)
  {
    cv::Mat pt  = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
    cv::Mat cam = R * pt + t;
    double x = cam.at<double>(0);
    double y = cam.at<double>(1);
    double z = cam.at<double>(2);
    if (z <= 0.1) continue;

    int u = static_cast<int>(fx * x / z + cx + 0.5);
    int v = static_cast<int>(fy * y / z + cy + 0.5);
    if (u < 0 || u >= w || v < 0 || v >= h) continue;

    float depth = std::min(static_cast<float>(z) / 20.0f, 1.0f);
    cv::Scalar color(255,
                     static_cast<int>(255 * (1.0 - depth)),
                     static_cast<int>(255 * depth));
    cv::circle(out, {u, v}, 2, color, -1);
  }
  return out;
}

}  // namespace multicamera_lidar_calibration

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<multicamera_lidar_calibration::MultiCameraSubscriber>());
  rclcpp::shutdown();
  return 0;
}