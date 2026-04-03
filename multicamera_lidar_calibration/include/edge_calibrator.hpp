#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <string>
#include <algorithm>

namespace multicamera_lidar_calibration
{

struct CalibFrame
{
  cv::Mat  image_edges;   // Canny edges on undistorted image (CV_8U)
  cv::Mat  distance_map;  // Distance transform of edge image (CV_32F)
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
};

// ─────────────────────────────────────────────────────────────────────────────
// Jet scalar extractor — works for both double and ceres::Jet<double,N>
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
inline double jetToDouble(const T& x) { return x.a; }
template <>
inline double jetToDouble<double>(const double& x) { return x; }

// ─────────────────────────────────────────────────────────────────────────────
// Ceres cost functor
// ─────────────────────────────────────────────────────────────────────────────
struct EdgeAlignCost
{
  EdgeAlignCost(const cv::Mat& dist_map,
                const cv::Mat& K,
                const pcl::PointXYZI& pt)
    : dist_map_(dist_map), pt_(pt)
  {
    fx_ = K.at<double>(0, 0);
    fy_ = K.at<double>(1, 1);
    cx_ = K.at<double>(0, 2);
    cy_ = K.at<double>(1, 2);
    w_  = dist_map.cols;
    h_  = dist_map.rows;
  }

  // Bilinear interpolation — index arithmetic uses scalar part only
  template <typename T>
  T interpolate(T u, T v) const
  {
    int x0 = static_cast<int>(std::floor(jetToDouble(u)));
    int y0 = static_cast<int>(std::floor(jetToDouble(v)));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = std::clamp(x0, 0, w_ - 1);
    x1 = std::clamp(x1, 0, w_ - 1);
    y0 = std::clamp(y0, 0, h_ - 1);
    y1 = std::clamp(y1, 0, h_ - 1);

    T dx = u - T(x0);
    T dy = v - T(y0);

    T f00 = T(dist_map_.at<float>(y0, x0));
    T f10 = T(dist_map_.at<float>(y0, x1));
    T f01 = T(dist_map_.at<float>(y1, x0));
    T f11 = T(dist_map_.at<float>(y1, x1));

    return (T(1) - dx) * (T(1) - dy) * f00
         + dx           * (T(1) - dy) * f10
         + (T(1) - dx)  * dy          * f01
         + dx            * dy          * f11;
  }

  template <typename T>
  bool operator()(const T* const angle_axis,
                  const T* const translation,
                  T* residual) const
  {
    T p_lidar[3] = { T(pt_.x), T(pt_.y), T(pt_.z) };
    T p_cam[3];
    ceres::AngleAxisRotatePoint(angle_axis, p_lidar, p_cam);
    p_cam[0] += translation[0];
    p_cam[1] += translation[1];
    p_cam[2] += translation[2];

    if (jetToDouble(p_cam[2]) < 0.1)
    {
      residual[0] = T(0);
      return true;
    }

    T u = T(fx_) * p_cam[0] / p_cam[2] + T(cx_);
    T v = T(fy_) * p_cam[1] / p_cam[2] + T(cy_);

    double u_s = jetToDouble(u);
    double v_s = jetToDouble(v);

    if (u_s < 0.0 || u_s >= static_cast<double>(w_) ||
        v_s < 0.0 || v_s >= static_cast<double>(h_))
    {
      residual[0] = T(0);
      return true;
    }

    residual[0] = interpolate(u, v);
    return true;
  }

  static ceres::CostFunction* Create(const cv::Mat& dist_map,
                                     const cv::Mat& K,
                                     const pcl::PointXYZI& pt)
  {
    return new ceres::AutoDiffCostFunction<EdgeAlignCost, 1, 3, 3>(
        new EdgeAlignCost(dist_map, K, pt));
  }

private:
  cv::Mat        dist_map_;
  pcl::PointXYZI pt_;
  double         fx_, fy_, cx_, cy_;
  int            w_, h_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main calibrator class
// ─────────────────────────────────────────────────────────────────────────────
class EdgeCalibrator
{
public:
  struct Config
  {
    int    canny_low          = 50;
    int    canny_high         = 150;
    int    min_frames         = 30;    // more frames = more residuals = robust
    int    max_frames         = 60;
    double dist_map_scale     = 1.0;
    double depth_edge_thresh  = 0.5;   // depth discontinuity threshold (m)
    int    lidar_sample_step  = 3;

    // ── Sanity checks on the solve result ──────────────────────────────────
    int    min_residuals      = 2000;  // reject solve if fewer residuals than this
    double max_translation_m  = 0.5;   // reject if t moves more than this (metres)
    double max_rotation_deg   = 10.0;  // reject if R changes more than this (degrees)
  };

  EdgeCalibrator();
  explicit EdgeCalibrator(const Config& cfg);

  bool addFrame(const cv::Mat& undistorted_image,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                const cv::Mat& K,
                const cv::Mat& R_init,
                const cv::Mat& t_init);

  bool hasEnoughFrames() const { return static_cast<int>(frames_.size()) >= cfg_.min_frames; }
  int  frameCount()      const { return static_cast<int>(frames_.size()); }

  bool solve(const cv::Mat& R_init,
             const cv::Mat& t_init,
             cv::Mat& R_out,
             cv::Mat& t_out,
             bool verbose = true);

  static cv::Mat extractImageEdges(const cv::Mat& undistorted,
                                   int low = 50, int high = 150);

  static cv::Mat buildDistanceMap(const cv::Mat& edges);

  static std::vector<pcl::PointXYZI>
  extractLidarEdgePoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                         const cv::Mat& R, const cv::Mat& t, const cv::Mat& K,
                         int w, int h, double depth_thresh = 0.5);

  void reset() { frames_.clear(); }

private:
  // Check that the refined result is physically plausible
  bool isSane(const cv::Mat& R_init, const cv::Mat& t_init,
              const cv::Mat& R_out,  const cv::Mat& t_out) const;

  Config                  cfg_;
  std::vector<CalibFrame> frames_;
  cv::Mat                 K_stored_;
};

}  // namespace multicamera_lidar_calibration