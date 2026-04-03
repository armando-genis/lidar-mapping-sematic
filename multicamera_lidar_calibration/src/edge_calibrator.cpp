#include "edge_calibrator.hpp"
#include <iostream>
#include <limits>

namespace multicamera_lidar_calibration
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructors
// ─────────────────────────────────────────────────────────────────────────────
EdgeCalibrator::EdgeCalibrator()
  : cfg_(Config{})
{}

EdgeCalibrator::EdgeCalibrator(const Config& cfg)
  : cfg_(cfg)
{}

// ─────────────────────────────────────────────────────────────────────────────
// Static helpers
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat EdgeCalibrator::extractImageEdges(const cv::Mat& undistorted,
                                          int low, int high)
{
  cv::Mat gray, blurred, edges;
  if (undistorted.channels() == 3)
    cv::cvtColor(undistorted, gray, cv::COLOR_BGR2GRAY);
  else
    gray = undistorted.clone();
  cv::GaussianBlur(gray, blurred, {5, 5}, 1.5);
  cv::Canny(blurred, edges, low, high);
  return edges;
}

cv::Mat EdgeCalibrator::buildDistanceMap(const cv::Mat& edges)
{
  cv::Mat inv;
  cv::bitwise_not(edges, inv);
  cv::Mat dist;
  cv::distanceTransform(inv, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
  double maxVal = 0.0;
  cv::minMaxLoc(dist, nullptr, &maxVal);
  if (maxVal > 0.0) dist /= static_cast<float>(maxVal);
  return dist;  // CV_32F, 0=on edge, 1=far
}

std::vector<pcl::PointXYZI>
EdgeCalibrator::extractLidarEdgePoints(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const cv::Mat& R, const cv::Mat& t, const cv::Mat& K,
    int w, int h, double depth_thresh)
{
  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);

  cv::Mat depth_img(h, w, CV_32F, cv::Scalar(0.f));

  struct PixelHit { int idx; float z; };
  std::vector<PixelHit> hit_map(static_cast<size_t>(w * h),
                                {-1, std::numeric_limits<float>::max()});

  for (int i = 0; i < static_cast<int>(cloud->size()); ++i)
  {
    const auto& p = cloud->points[i];
    cv::Mat pt  = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
    cv::Mat cam = R * pt + t;
    double z    = cam.at<double>(2);
    if (z < 0.1) continue;
    int u = static_cast<int>(fx * cam.at<double>(0) / z + cx + 0.5);
    int v = static_cast<int>(fy * cam.at<double>(1) / z + cy + 0.5);
    if (u < 0 || u >= w || v < 0 || v >= h) continue;

    auto& hit = hit_map[static_cast<size_t>(v * w + u)];
    if (static_cast<float>(z) < hit.z)
    {
      hit.z   = static_cast<float>(z);
      hit.idx = i;
      depth_img.at<float>(v, u) = static_cast<float>(z);
    }
  }

  cv::Mat grad_x, grad_y, grad_mag;
  cv::Sobel(depth_img, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(depth_img, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, grad_mag);

  std::vector<pcl::PointXYZI> edge_pts;
  for (int v = 0; v < h; ++v)
    for (int u = 0; u < w; ++u)
    {
      const auto& hit = hit_map[static_cast<size_t>(v * w + u)];
      if (hit.idx < 0) continue;
      if (grad_mag.at<float>(v, u) > static_cast<float>(depth_thresh))
        edge_pts.push_back(cloud->points[hit.idx]);
    }

  return edge_pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// isSane — rejects physically implausible optimisation results
// ─────────────────────────────────────────────────────────────────────────────
bool EdgeCalibrator::isSane(const cv::Mat& R_init, const cv::Mat& t_init,
                             const cv::Mat& R_out,  const cv::Mat& t_out) const
{
  // ── Translation change ────────────────────────────────────────────────────
  cv::Mat dt = t_out - t_init;
  double t_change = cv::norm(dt);
  if (t_change > cfg_.max_translation_m)
  {
    std::cerr << "[EdgeCalibrator] SANITY FAIL: translation moved "
              << t_change << " m (max " << cfg_.max_translation_m << " m)\n";
    return false;
  }

  // ── Rotation change (as angle of R_out * R_init^T) ────────────────────────
  cv::Mat dR = R_out * R_init.t();
  cv::Mat rvec;
  cv::Rodrigues(dR, rvec);
  double angle_deg = cv::norm(rvec) * 180.0 / CV_PI;
  if (angle_deg > cfg_.max_rotation_deg)
  {
    std::cerr << "[EdgeCalibrator] SANITY FAIL: rotation changed "
              << angle_deg << " deg (max " << cfg_.max_rotation_deg << " deg)\n";
    return false;
  }

  std::cout << "[EdgeCalibrator] Sanity OK: dt=" << t_change
            << " m, dR=" << angle_deg << " deg\n";
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// addFrame
// ─────────────────────────────────────────────────────────────────────────────
bool EdgeCalibrator::addFrame(
    const cv::Mat& undistorted_image,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
    const cv::Mat& K,
    const cv::Mat& /*R_init*/,
    const cv::Mat& /*t_init*/)
{
  if (static_cast<int>(frames_.size()) >= cfg_.max_frames)
  {
    std::cout << "[EdgeCalibrator] Max frames reached, skipping.\n";
    return false;
  }

  K_stored_ = K.clone();

  CalibFrame f;
  f.cloud        = cloud;
  f.image_edges  = extractImageEdges(undistorted_image, cfg_.canny_low, cfg_.canny_high);
  f.distance_map = buildDistanceMap(f.image_edges);

  frames_.push_back(std::move(f));
  std::cout << "[EdgeCalibrator] Frame added: "
            << frames_.size() << "/" << cfg_.max_frames << "\n";
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// solve
// ─────────────────────────────────────────────────────────────────────────────
bool EdgeCalibrator::solve(
    const cv::Mat& R_init,
    const cv::Mat& t_init,
    cv::Mat& R_out,
    cv::Mat& t_out,
    bool verbose)
{
  if (!hasEnoughFrames())
  {
    std::cerr << "[EdgeCalibrator] Not enough frames ("
              << frames_.size() << "/" << cfg_.min_frames << ")\n";
    return false;
  }

  // ── Convert initial R to axis-angle ──────────────────────────────────────
  double angle_axis[3];
  {
    cv::Mat rvec;
    cv::Rodrigues(R_init, rvec);
    angle_axis[0] = rvec.at<double>(0);
    angle_axis[1] = rvec.at<double>(1);
    angle_axis[2] = rvec.at<double>(2);
  }

  double translation[3] = {
    t_init.at<double>(0),
    t_init.at<double>(1),
    t_init.at<double>(2)
  };

  // ── Build problem ─────────────────────────────────────────────────────────
  ceres::Problem problem;
  int total_residuals = 0;

  for (auto& frame : frames_)
  {
    // Use current estimate to project lidar and extract edge points
    cv::Mat R_curr;
    cv::Mat rvec_curr = (cv::Mat_<double>(3, 1)
                          << angle_axis[0], angle_axis[1], angle_axis[2]);
    cv::Rodrigues(rvec_curr, R_curr);
    cv::Mat t_curr = (cv::Mat_<double>(3, 1)
                       << translation[0], translation[1], translation[2]);

    auto edge_pts = extractLidarEdgePoints(
        frame.cloud, R_curr, t_curr, K_stored_,
        frame.distance_map.cols, frame.distance_map.rows,
        cfg_.depth_edge_thresh);

    for (int i = 0; i < static_cast<int>(edge_pts.size()); i += cfg_.lidar_sample_step)
    {
      problem.AddResidualBlock(
          EdgeAlignCost::Create(frame.distance_map, K_stored_, edge_pts[i]),
          new ceres::HuberLoss(0.5),  // robust loss to ignore outliers
          angle_axis,
          translation);
      ++total_residuals;
    }
  }

  if (verbose)
    std::cout << "[EdgeCalibrator] Total residuals: " << total_residuals << "\n";

  // ── Guard: reject if too few residuals (underconstrained) ────────────────
  if (total_residuals < cfg_.min_residuals)
  {
    std::cerr << "[EdgeCalibrator] REJECTED: only " << total_residuals
              << " residuals (need " << cfg_.min_residuals
              << "). Calibration is underconstrained — skipping this batch.\n"
              << "  → Check that the LiDAR has enough points in the camera FOV\n"
              << "  → Consider lowering depth_edge_thresh or lidar_sample_step\n";
    return false;
  }

  // ── Solver options ────────────────────────────────────────────────────────
  ceres::Solver::Options options;
  options.linear_solver_type           = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = verbose;
  options.max_num_iterations           = 200;
  options.function_tolerance           = 1e-8;
  options.gradient_tolerance           = 1e-10;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  if (verbose)
    std::cout << summary.FullReport() << "\n";

  if (summary.termination_type == ceres::FAILURE)
  {
    std::cerr << "[EdgeCalibrator] Ceres returned FAILURE.\n";
    return false;
  }

  // ── Convert back ─────────────────────────────────────────────────────────
  cv::Mat rvec_out = (cv::Mat_<double>(3, 1)
                       << angle_axis[0], angle_axis[1], angle_axis[2]);
  cv::Rodrigues(rvec_out, R_out);
  t_out = (cv::Mat_<double>(3, 1)
            << translation[0], translation[1], translation[2]);

  if (verbose)
  {
    std::cout << "\n[EdgeCalibrator] === Refined extrinsics ===\n";
    std::cout << "R:\n" << R_out << "\n";
    std::cout << "t:\n" << t_out << "\n";
  }

  // ── Sanity check ──────────────────────────────────────────────────────────
  if (!isSane(R_init, t_init, R_out, t_out))
  {
    std::cerr << "[EdgeCalibrator] Result failed sanity check — discarding.\n";
    return false;
  }

  return true;
}

}  // namespace multicamera_lidar_calibration