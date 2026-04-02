#include "edge_calibrator.hpp"

#include <iostream>
#include <limits>

namespace multicamera_lidar_calibration
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructors  (defined here so Config{} is fully known)
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
  // Invert: edge pixels → 0, background → 255  (distanceTransform needs 0 = obstacle)
  cv::Mat inv;
  cv::bitwise_not(edges, inv);

  cv::Mat dist;
  cv::distanceTransform(inv, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

  // Normalise to [0, 1] so far-from-edge = 1 (high cost), on-edge = 0 (zero cost)
  double maxVal = 0.0;
  cv::minMaxLoc(dist, nullptr, &maxVal);
  if (maxVal > 0.0)
    dist /= static_cast<float>(maxVal);

  return dist;  // CV_32F
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

  // ── Build a depth image by projecting the cloud ───────────────────────────
  cv::Mat depth_img(h, w, CV_32F, cv::Scalar(0.f));

  // We also need to remember which cloud index maps to each pixel
  // (keep closest point per pixel)
  struct PixelHit { int idx; float z; };
  std::vector<PixelHit> hit_map(static_cast<size_t>(w * h), {-1, std::numeric_limits<float>::max()});

  for (int i = 0; i < static_cast<int>(cloud->size()); ++i)
  {
    const auto& p = cloud->points[i];
    cv::Mat pt = (cv::Mat_<double>(3, 1) << p.x, p.y, p.z);
    cv::Mat cam = R * pt + t;

    double z = cam.at<double>(2);
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

  // ── Depth-gradient → find discontinuities ────────────────────────────────
  cv::Mat grad_x, grad_y, grad_mag;
  cv::Sobel(depth_img, grad_x, CV_32F, 1, 0, 3);
  cv::Sobel(depth_img, grad_y, CV_32F, 0, 1, 3);
  cv::magnitude(grad_x, grad_y, grad_mag);

  // ── Collect cloud points that project onto depth edges ───────────────────
  std::vector<pcl::PointXYZI> edge_pts;
  for (int v = 0; v < h; ++v)
  {
    for (int u = 0; u < w; ++u)
    {
      const auto& hit = hit_map[static_cast<size_t>(v * w + u)];
      if (hit.idx < 0) continue;
      if (grad_mag.at<float>(v, u) > static_cast<float>(depth_thresh))
        edge_pts.push_back(cloud->points[hit.idx]);
    }
  }

  return edge_pts;
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
    std::cout << "[EdgeCalibrator] Max frames reached (" << cfg_.max_frames << "), skipping.\n";
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

  // ── Convert initial R (3×3) → axis-angle ─────────────────────────────────
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

  // ── Build Ceres problem ───────────────────────────────────────────────────
  ceres::Problem problem;
  int total_residuals = 0;

  for (auto& frame : frames_)
  {
    // Recompute current R, t from optimisation variables for edge extraction
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
          nullptr,       // no robust loss — add ceres::HuberLoss(1.0) if needed
          angle_axis,
          translation
      );
      ++total_residuals;
    }
  }

  if (verbose)
    std::cout << "[EdgeCalibrator] Total residuals: " << total_residuals << "\n";

  if (total_residuals == 0)
  {
    std::cerr << "[EdgeCalibrator] No residuals — check depth_edge_thresh or cloud validity.\n";
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
    return false;

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

  return true;
}

}  // namespace multicamera_lidar_calibration