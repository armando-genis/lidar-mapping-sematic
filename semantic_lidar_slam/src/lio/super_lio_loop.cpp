
#include "lio/super_lio_loop.h"
#include "basic/logs.h"
#include <opencv2/highgui.hpp>

using namespace BASIC;

namespace LI2Sup {

// ============================================================================
//  Constructor / Destructor
// ============================================================================

SuperLIOLoop::SuperLIOLoop()
{
  kf_positions_.reset(new pcl::PointCloud<pcl::PointXYZ>());
}

SuperLIOLoop::~SuperLIOLoop()
{
  // Signal the loop thread to stop and wait for it to finish.
  loop_thread_stop_.store(true);
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
  delete isam_;
}

// ============================================================================
//  init()  – runs once before the main processing loop starts
// ============================================================================

void SuperLIOLoop::init()
{
  // Delegate base initialisation (ESKF, iVox, etc.)
  SuperLIO::init();

  if (!g_loop_closure_enable && !g_local_map_enable) return;

  // ── ROS2 publishers — always created when any feature is active ───────────
  pub_corrected_path_ = data_wrapper_->create_publisher<nav_msgs::msg::Path>(
    "/super_lio/corrected_path", 10);
  pub_loop_constraints_ = data_wrapper_->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/super_lio/loop_closure_constraints", 1);

  LOG(INFO) << GREEN << " ---> [SuperLIOLoop]: Path publisher active on /super_lio/corrected_path" << RESET;

  if (!g_loop_closure_enable) {
    LOG(INFO) << YELLOW << " ---> [SuperLIOLoop]: Loop closure DISABLED — running in local-map odometry mode." << RESET;
    return;
  }

  // ── iSAM2 ─────────────────────────────────────────────────────────────────
  gtsam::ISAM2Params isam_params;
  isam_params.relinearizeThreshold = 0.01;
  isam_params.relinearizeSkip      = 1;
  isam_ = new gtsam::ISAM2(isam_params);

  // ── ICP local-map voxel filter ────────────────────────────────────────────
  icp_ds_filter_.setLeafSize(
    static_cast<float>(g_loop_icp_leaf_size),
    static_cast<float>(g_loop_icp_leaf_size),
    static_cast<float>(g_loop_icp_leaf_size));

  // ── Background loop-closure thread ───────────────────────────────────────
  loop_thread_ = std::thread(&SuperLIOLoop::loopClosureThread, this);

  LOG(INFO) << GREEN << " ---> [SuperLIOLoop]: Loop closure ENABLED." << RESET;
}

// ============================================================================
//  UpdateMap()  – extends base: saves keyframes after map update
// ============================================================================

void SuperLIOLoop::UpdateMap()
{
  // Let the base class insert the current scan into iVox and update last_pose_.
  SuperLIO::UpdateMap();

  if (!g_loop_closure_enable && !g_local_map_enable) return;
  if (ds_undistort_->empty()) return;

  if (shouldAddKeyframe()) {
    addKeyframe();
    new_keyframe_ = true;
    prune_counter_++;
  }
}

// ============================================================================
//  Output()  – extends base: drives the factor graph and map rebuild
// ============================================================================

void SuperLIOLoop::Output()
{
  // Publish odometry / point-cloud as usual.
  SuperLIO::Output();

  // ── Show synced camera frame if available ──────────────────────────────
  if (!measures_.image_frame.empty()) {
    cv::imshow("Camera", measures_.image_frame);
    cv::waitKey(1);
  }

  if (!g_loop_closure_enable && !g_local_map_enable) return;

  if (new_keyframe_) {
    new_keyframe_ = false;

    // ── Loop closure: drive the GTSAM factor graph ────────────────────────
    if (g_loop_closure_enable) {
      addOdomFactor();
      addLoopFactor();

      bool loop_closed = flag_rebuild_.load();
      optimizeGraph(loop_closed);

      if (loop_closed) {
        // rebuildMap already clears iVox with corrected poses — no separate prune needed.
        rebuildMap();
        flag_rebuild_.store(false);
        prune_counter_ = 0;
        publishLoopConstraints();
        LOG(INFO) << GREEN << " ---> [SuperLIOLoop]: Map rebuilt after loop closure." << RESET;
      }
    }

    // ── Local sliding-window prune (runs independently of loop closure) ────
    if (g_local_map_enable && prune_counter_ >= g_local_map_prune_interval) {
      pruneLocalMap();
      prune_counter_ = 0;
    }

    // ── Publish live trajectory on every keyframe ──────────────────────────
    publishCorrectedPath();
  }
}

// ============================================================================
//  Keyframe management
// ============================================================================

bool SuperLIOLoop::shouldAddKeyframe() const
{
  if (kf_count_ == 0) return true;   // always save the very first keyframe

  float dist = (last_pose_.t_ - last_kf_se3_.t_).norm();
  if (dist < static_cast<float>(g_loop_keyframe_add_dist)) return false;

  // Optional angle check
  M3 dR = last_kf_se3_.R_.transpose() * last_pose_.R_;
  float angle = Eigen::AngleAxisf(dR).angle();
  if (dist < static_cast<float>(g_loop_keyframe_add_dist * 0.5f) &&
      angle < static_cast<float>(g_loop_keyframe_add_angle)) {
    return false;
  }

  return true;
}

void SuperLIOLoop::addKeyframe()
{
  KeyFrame kf;
  kf.pose      = last_pose_;
  kf.timestamp = measures_.lidar.end_time;

  // Copy the body-frame downsampled scan so we can re-project it later.
  kf.cloud.reset(new PointCloudType());
  *kf.cloud = *ds_undistort_;

  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    keyframes_.push_back(kf);

    pcl::PointXYZ kf_pos;
    kf_pos.x = static_cast<float>(last_pose_.t_(0));
    kf_pos.y = static_cast<float>(last_pose_.t_(1));
    kf_pos.z = static_cast<float>(last_pose_.t_(2));
    kf_positions_->push_back(kf_pos);
  }

  last_kf_se3_ = last_pose_;
}

// ============================================================================
//  GTSAM factor graph
// ============================================================================

void SuperLIOLoop::addOdomFactor()
{
  gtsam::Pose3 cur_gtsam = toGtsamPose(last_kf_se3_);

  // Odometry noise: tight on rotation, relaxed on translation (tunable).
  gtsam::Vector6 odom_noise_vec;
  odom_noise_vec << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
  auto odom_noise = gtsam::noiseModel::Diagonal::Variances(odom_noise_vec);

  if (kf_count_ == 0) {
    // Prior factor anchors the very first node to the world origin.
    gtsam::Vector6 prior_noise_vec;
    prior_noise_vec << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    auto prior_noise = gtsam::noiseModel::Diagonal::Variances(prior_noise_vec);

    gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, cur_gtsam, prior_noise));
    gtsam_init_vals_.insert(0, cur_gtsam);

  } else {
    // BetweenFactor: relative transform from previous keyframe to current.
    gtsam::Pose3 relative = last_kf_gtsam_.between(cur_gtsam);
    gtsam_graph_.add(
      gtsam::BetweenFactor<gtsam::Pose3>(kf_count_ - 1, kf_count_, relative, odom_noise));
    gtsam_init_vals_.insert(kf_count_, cur_gtsam);
  }

  last_kf_gtsam_ = cur_gtsam;
  has_last_kf_   = true;
  kf_count_++;
}

void SuperLIOLoop::addLoopFactor()
{
  std::lock_guard<std::mutex> lock(mtx_loop_);
  if (loop_index_queue_.empty()) return;

  for (size_t i = 0; i < loop_index_queue_.size(); ++i) {
    int idx_from = loop_index_queue_[i].first;
    int idx_to   = loop_index_queue_[i].second;
    gtsam_graph_.add(
      gtsam::BetweenFactor<gtsam::Pose3>(
        idx_from, idx_to,
        loop_pose_queue_[i],
        loop_noise_queue_[i]));
  }

  loop_index_queue_.clear();
  loop_pose_queue_.clear();
  loop_noise_queue_.clear();
}

void SuperLIOLoop::optimizeGraph(bool loop_closed)
{
  isam_->update(gtsam_graph_, gtsam_init_vals_);
  isam_->update();

  // Extra iterations to fully propagate a loop closure through the graph.
  if (loop_closed) {
    for (int i = 0; i < 5; ++i) isam_->update();
  }

  gtsam_graph_.resize(0);
  gtsam_init_vals_.clear();

  isam_estimate_ = isam_->calculateEstimate();
}

// ============================================================================
//  Map rebuild
//  Called on the main thread after a loop closure has been accepted and the
//  graph has been re-optimised.  Steps:
//    1. Fetch all corrected poses from iSAM2.
//    2. Update stored keyframe poses and the KD-tree position cloud.
//    3. Reset iVox and re-insert every keyframe cloud at its corrected pose.
//    4. Correct the live ESKF state to match the corrected current pose.
// ============================================================================

void SuperLIOLoop::rebuildMap()
{
  // ── 1. Corrected poses from iSAM2 ────────────────────────────────────────
  gtsam::Values corrected = isam_->calculateEstimate();

  // ── 2 & 3. Re-insert keyframe clouds into a fresh iVox ───────────────────
  ivox_.reset(new OctVoxMapType(OctVoxMapType::Options{g_ivox_resolution, g_ivox_capacity}));

  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    kf_positions_->clear();

    for (int i = 0; i < kf_count_; ++i) {
      if (!corrected.exists(i)) continue;

      gtsam::Pose3 corrected_pose = corrected.at<gtsam::Pose3>(i);
      SE3 corrected_se3           = fromGtsamPose(corrected_pose);

      // Update stored pose so the loop thread always reads fresh values.
      keyframes_[i].pose = corrected_se3;

      // Update KD-tree position cloud.
      pcl::PointXYZ p;
      p.x = static_cast<float>(corrected_se3.t_(0));
      p.y = static_cast<float>(corrected_se3.t_(1));
      p.z = static_cast<float>(corrected_se3.t_(2));
      kf_positions_->push_back(p);

      // Re-project body-frame cloud into world frame and insert into iVox.
      const auto& cloud = keyframes_[i].cloud;
      VV3 world_pts;
      world_pts.reserve(cloud->size());
      for (const auto& pt : cloud->points) {
        V3 p_body(pt.x, pt.y, pt.z);
        world_pts.push_back(corrected_se3 * p_body);
      }
      ivox_->insert(world_pts);
    }
  }

  // ── 4. Correct the live ESKF state ───────────────────────────────────────
  if (corrected.exists(kf_count_ - 1)) {
    gtsam::Pose3 corrected_cur = corrected.at<gtsam::Pose3>(kf_count_ - 1);
    SE3 corrected_se3          = fromGtsamPose(corrected_cur);

    auto sys_state = kf_->GetSysState();
    sys_state.R    = SO3(corrected_se3.R_);
    sys_state.p    = corrected_se3.t_;
    kf_->SetX(sys_state);

    last_kf_se3_    = corrected_se3;
    last_kf_gtsam_  = toGtsamPose(corrected_se3);
    last_pose_      = corrected_se3;
  }
}

// ============================================================================
//  Loop closure – background thread
// ============================================================================

void SuperLIOLoop::loopClosureThread()
{
  rclcpp::Rate rate(g_loop_closure_frequency);

  while (!loop_thread_stop_.load()) {
    rate.sleep();

    {
      // Need at least one keyframe before attempting detection.
      std::lock_guard<std::mutex> lock(mtx_kf_);
      if (keyframes_.empty()) continue;
    }

    performLoopClosure();
  }
}

bool SuperLIOLoop::performLoopClosure()
{
  int cur_id = -1;
  int pre_id = -1;

  if (!detectLoop(&cur_id, &pre_id)) return false;

  // ── Build local maps ──────────────────────────────────────────────────────
  CloudPtr cur_cloud(new PointCloudType());
  CloudPtr pre_cloud(new PointCloudType());

  buildLocalMap(cur_cloud, cur_id, 0);                      // current: single keyframe
  buildLocalMap(pre_cloud, pre_id, g_loop_search_num);      // previous: wide neighbourhood

  if (cur_cloud->size() < 300 || pre_cloud->size() < 1000) {
    LOG(WARNING) << YELLOW << " ---> [Loop]: Candidate clouds too small ("
                 << cur_cloud->size() << " / " << pre_cloud->size() << "), skipped." << RESET;
    return false;
  }

  // ── ICP alignment ─────────────────────────────────────────────────────────
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(g_loop_search_radius * 2.0);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  icp.setInputSource(cur_cloud);
  icp.setInputTarget(pre_cloud);
  CloudPtr unused(new PointCloudType());
  icp.align(*unused);

  if (!icp.hasConverged() || icp.getFitnessScore() > g_loop_fitness_score) {
    LOG(INFO) << YELLOW << " ---> [Loop]: ICP rejected (converged="
              << icp.hasConverged()
              << ", score=" << icp.getFitnessScore() << ")." << RESET;
    return false;
  }

  LOG(INFO) << GREEN << " ---> [Loop]: Accepted! cur=" << cur_id
            << " pre=" << pre_id
            << " score=" << icp.getFitnessScore() << RESET;

  // ── Compute pose correction ───────────────────────────────────────────────
  SE3 cur_pose, pre_pose;
  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    cur_pose = keyframes_[cur_id].pose;
    pre_pose = keyframes_[pre_id].pose;
  }

  Eigen::Affine3f T_correction(icp.getFinalTransformation());
  Eigen::Affine3f T_wrong    = Eigen::Affine3f::Identity();
  T_wrong.linear()           = cur_pose.R_;
  T_wrong.translation()      = cur_pose.t_;
  Eigen::Affine3f T_correct  = T_correction * T_wrong;

  float x, y, z, roll, pitch, yaw;
  pcl::getTranslationAndEulerAngles(T_correct, x, y, z, roll, pitch, yaw);

  gtsam::Pose3 pose_from(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
                          gtsam::Point3(x, y, z));
  gtsam::Pose3 pose_to = toGtsamPose(pre_pose);

  float ns = static_cast<float>(icp.getFitnessScore());
  gtsam::Vector6 noise_vec;
  noise_vec << ns, ns, ns, ns, ns, ns;
  auto noise = gtsam::noiseModel::Diagonal::Variances(noise_vec);

  // ── Push to queues (consumed by main thread in addLoopFactor) ─────────────
  {
    std::lock_guard<std::mutex> lock(mtx_loop_);
    loop_index_queue_.emplace_back(cur_id, pre_id);
    loop_pose_queue_.push_back(pose_from.between(pose_to));
    loop_noise_queue_.push_back(noise);
    loop_history_[cur_id] = pre_id;
  }

  flag_rebuild_.store(true);
  return true;
}

bool SuperLIOLoop::detectLoop(int* curID, int* preID)
{
  // Take a snapshot of keyframe positions to avoid holding the lock during search.
  pcl::PointCloud<pcl::PointXYZ>::Ptr positions_copy(new pcl::PointCloud<pcl::PointXYZ>());
  int n_kf;
  double latest_stamp;
  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    if (keyframes_.size() < 2) return false;
    *positions_copy = *kf_positions_;
    n_kf            = static_cast<int>(keyframes_.size());
    latest_stamp    = keyframes_.back().timestamp;
  }

  int loop_key_cur = n_kf - 1;

  // Avoid re-detecting a loop we already found for this keyframe.
  {
    std::lock_guard<std::mutex> lock(mtx_loop_);
    if (loop_history_.count(loop_key_cur)) return false;
  }

  // KD-tree radius search over all keyframe positions.
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(positions_copy);

  std::vector<int>   indices;
  std::vector<float> sq_dists;
  kdtree.radiusSearch(
    positions_copy->points[loop_key_cur],
    g_loop_search_radius, indices, sq_dists);

  int loop_key_pre = -1;
  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    for (int idx : indices) {
      double stamp_diff = latest_stamp - keyframes_[idx].timestamp;
      if (stamp_diff > g_loop_search_time_diff) {
        loop_key_pre = idx;
        break;
      }
    }
  }

  if (loop_key_pre == -1 || loop_key_pre == loop_key_cur) return false;

  *curID = loop_key_cur;
  *preID = loop_key_pre;
  return true;
}

void SuperLIOLoop::buildLocalMap(BASIC::CloudPtr& out, int centerIdx, int halfWidth)
{
  out.reset(new PointCloudType());

  std::lock_guard<std::mutex> lock(mtx_kf_);
  int n = static_cast<int>(keyframes_.size());

  for (int i = centerIdx - halfWidth; i <= centerIdx + halfWidth; ++i) {
    if (i < 0 || i >= n) continue;

    const auto& kf = keyframes_[i];
    for (const auto& pt : kf.cloud->points) {
      V3 p_body(pt.x, pt.y, pt.z);
      V3 p_world = kf.pose * p_body;

      PointType p_out;
      p_out.x         = p_world(0);
      p_out.y         = p_world(1);
      p_out.z         = p_world(2);
      p_out.intensity = pt.intensity;
      out->push_back(p_out);
    }
  }

  // Downsample to keep ICP fast.
  if (!out->empty()) {
    CloudPtr tmp(new PointCloudType());
    icp_ds_filter_.setInputCloud(out);
    icp_ds_filter_.filter(*tmp);
    out = tmp;
  }
}

// ============================================================================
//  Sliding-window local map prune
//
//  Clears the iVox and re-inserts only keyframes whose position is within
//  g_local_map_radius metres of the current robot pose.  This keeps RAM and
//  KNN search cost bounded regardless of how long the robot has been running.
// ============================================================================

void SuperLIOLoop::pruneLocalMap()
{
  const V3    cur_pos    = last_pose_.t_;
  const float radius_sq  = static_cast<float>(g_local_map_radius * g_local_map_radius);

  // Fresh iVox — same resolution and capacity as the original.
  ivox_.reset(new OctVoxMapType(OctVoxMapType::Options{g_ivox_resolution, g_ivox_capacity}));

  int kept = 0;
  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    for (const auto& kf : keyframes_) {
      if ((kf.pose.t_ - cur_pos).squaredNorm() > radius_sq) continue;

      VV3 world_pts;
      world_pts.reserve(kf.cloud->size());
      for (const auto& pt : kf.cloud->points) {
        V3 p_body(pt.x, pt.y, pt.z);
        world_pts.push_back(kf.pose * p_body);
      }
      ivox_->insert(world_pts);
      kept++;
    }
  }

  LOG(INFO) << GREEN << " ---> [SuperLIOLoop]: Local map pruned — keeping "
            << kept << " / " << keyframes_.size()
            << " keyframes within " << g_local_map_radius << " m." << RESET;
}

// ============================================================================
//  Helpers
// ============================================================================

gtsam::Pose3 SuperLIOLoop::toGtsamPose(const BASIC::SE3& se3) const
{
  Eigen::Matrix3d R = se3.R_.cast<double>();
  Eigen::Vector3d t = se3.t_.cast<double>();
  return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t));
}

BASIC::SE3 SuperLIOLoop::fromGtsamPose(const gtsam::Pose3& pose) const
{
  M3 R = pose.rotation().matrix().cast<scalar>();
  V3 t = pose.translation().cast<scalar>();
  return SE3(SO3(R), t);
}

void SuperLIOLoop::publishLoopConstraints()
{
  if (!pub_loop_constraints_) return;

  std::lock_guard<std::mutex> lock(mtx_kf_);
  int n = static_cast<int>(keyframes_.size());
  if (n == 0) return;

  visualization_msgs::msg::MarkerArray marker_array;
  auto stamp = data_wrapper_->now();

  // Node spheres
  visualization_msgs::msg::Marker nodes;
  nodes.header.frame_id = "odom";
  nodes.header.stamp    = stamp;
  nodes.action          = visualization_msgs::msg::Marker::ADD;
  nodes.type            = visualization_msgs::msg::Marker::SPHERE_LIST;
  nodes.ns              = "loop_nodes";
  nodes.id              = 0;
  nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.3;
  nodes.color.r = 0.0f; nodes.color.g = 0.8f;
  nodes.color.b = 1.0f; nodes.color.a = 1.0f;

  // Edge lines
  visualization_msgs::msg::Marker edges;
  edges.header   = nodes.header;
  edges.action   = visualization_msgs::msg::Marker::ADD;
  edges.type     = visualization_msgs::msg::Marker::LINE_LIST;
  edges.ns       = "loop_edges";
  edges.id       = 1;
  edges.scale.x  = 0.1;
  edges.color.r  = 0.9f; edges.color.g = 0.0f;
  edges.color.b  = 0.9f; edges.color.a = 1.0f;

  {
    std::lock_guard<std::mutex> lk(mtx_loop_);
    for (const auto& [from, to] : loop_history_) {
      if (from >= n || to >= n) continue;

      geometry_msgs::msg::Point pf, pt_msg;
      pf.x = keyframes_[from].pose.t_(0);
      pf.y = keyframes_[from].pose.t_(1);
      pf.z = keyframes_[from].pose.t_(2);

      pt_msg.x = keyframes_[to].pose.t_(0);
      pt_msg.y = keyframes_[to].pose.t_(1);
      pt_msg.z = keyframes_[to].pose.t_(2);

      nodes.points.push_back(pf);
      nodes.points.push_back(pt_msg);
      edges.points.push_back(pf);
      edges.points.push_back(pt_msg);
    }
  }

  marker_array.markers.push_back(nodes);
  marker_array.markers.push_back(edges);
  pub_loop_constraints_->publish(marker_array);
}

void SuperLIOLoop::publishCorrectedPath()
{
  if (!pub_corrected_path_) return;

  nav_msgs::msg::Path path;
  path.header.frame_id = "odom";
  path.header.stamp    = data_wrapper_->now();

  std::lock_guard<std::mutex> lock(mtx_kf_);
  for (const auto& kf : keyframes_) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = path.header;
    ps.pose.position.x    = kf.pose.t_(0);
    ps.pose.position.y    = kf.pose.t_(1);
    ps.pose.position.z    = kf.pose.t_(2);

    Quat q(kf.pose.R_);
    ps.pose.orientation.w = q.w();
    ps.pose.orientation.x = q.x();
    ps.pose.orientation.y = q.y();
    ps.pose.orientation.z = q.z();

    path.poses.push_back(ps);
  }

  pub_corrected_path_->publish(path);
}

}  // namespace LI2Sup
