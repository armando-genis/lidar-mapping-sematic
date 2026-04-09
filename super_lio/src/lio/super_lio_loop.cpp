
#include "lio/super_lio_loop.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

using namespace BASIC;

namespace LI2Sup {

// ============================================================================
// Lifecycle
// ============================================================================

SuperLIOLoop::~SuperLIOLoop()
{
  loop_thread_running_ = false;
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}


void SuperLIOLoop::init()
{
  // Base class initialises ivox_, kf_, voxel filter, state machine, etc.
  SuperLIO::init();

  if (!g_loop_closure_enable) {
    LOG(INFO) << YELLOW << " ---> [LoopClosure]: disabled by config." << RESET;
    return;
  }

  // Create ROS2 publishers on the wrapper node.
  pub_loop_markers_ = data_wrapper_->create_publisher<
      visualization_msgs::msg::MarkerArray>("/lio/loop_closure_constraints", 10);

  pub_history_cloud_ = data_wrapper_->create_publisher<
      sensor_msgs::msg::PointCloud2>("/lio/loop_closure_history_cloud", 5);

  pub_icp_cloud_ = data_wrapper_->create_publisher<
      sensor_msgs::msg::PointCloud2>("/lio/loop_closure_icp_cloud", 5);

  loop_thread_running_ = true;
  loop_thread_ = std::thread(&SuperLIOLoop::loopClosureThread, this);

  LOG(INFO) << GREEN << " ---> [LoopClosure]: initialised. frequency="
            << g_loop_closure_frequency << " Hz, "
            << "search_radius=" << g_history_keyframe_search_radius << " m."
            << RESET;
}


// ============================================================================
// stateProcess override
// ============================================================================

void SuperLIOLoop::stateProcess()
{
  // 1. Apply any pending loop correction BEFORE the next scan-to-map step so
  //    the corrected map is used immediately.
  applyLoopCorrection();

  // 2. Run the full LIO pipeline (undistort → downsample → observe → update → output).
  SuperLIO::stateProcess();

  if (!g_loop_closure_enable) return;

  // 3. Save a keyframe if the robot moved / rotated enough.
  const SE3 cur_pose = kf_->GetSE3();
  if (shouldAddKeyframe(cur_pose)) {
    saveKeyframe(measures_.lidar.end_time, cur_pose);
  }
}


// ============================================================================
// Keyframe management
// ============================================================================

bool SuperLIOLoop::shouldAddKeyframe(const SE3& cur_pose) const
{
  if (!has_last_keyframe_pose_) return true;

  // Distance gate
  const V3 dp = cur_pose.t_ - last_keyframe_pose_.t_;
  if (dp.norm() >= static_cast<scalar>(g_surrounding_keyframe_adding_dist_threshold))
    return true;

  // Rotation gate — angle from rotation matrix trace: angle = acos((tr(R)-1)/2)
  const M3    R_delta = cur_pose.R_ * last_keyframe_pose_.R_.transpose();
  const float trace   = R_delta.trace();
  const float angle   = std::acos(std::clamp((trace - 1.0f) / 2.0f, -1.0f, 1.0f));
  if (angle >= static_cast<scalar>(g_surrounding_keyframe_adding_angle_threshold))
    return true;

  return false;
}


void SuperLIOLoop::saveKeyframe(double timestamp, const SE3& pose)
{
  KeyFrame kf;
  kf.timestamp = timestamp;
  kf.pose      = pose;

  // Deep-copy the current downsampled body-frame scan.
  kf.cloud.reset(new PointCloudType());
  *kf.cloud = *ds_undistort_;

  {
    std::lock_guard<std::mutex> lock(mtx_keyframes_);
    kf.id = static_cast<int>(keyframes_.size());
    keyframes_.push_back(std::move(kf));
  }

  last_keyframe_pose_     = pose;
  has_last_keyframe_pose_ = true;
}


// ============================================================================
// Loop closure background thread
// ============================================================================

void SuperLIOLoop::loopClosureThread()
{
  const auto period =
      std::chrono::duration<double>(1.0 / g_loop_closure_frequency);

  while (loop_thread_running_) {
    std::this_thread::sleep_for(period);
    performLoopClosure();
    visualizeLoopClosure();
  }
}


// ============================================================================
// Detection — radius search in keyframe position history
// ============================================================================

bool SuperLIOLoop::detectLoopClosureDistance(int* cur_id, int* pre_id)
{
  // --- Step 1: get latest keyframe index and its position (brief lock) ---
  int     latest;
  V3      cur_pos;
  double  cur_time;

  {
    std::lock_guard<std::mutex> lock(mtx_keyframes_);
    if (static_cast<int>(keyframes_.size()) < 2) return false;
    latest   = static_cast<int>(keyframes_.size()) - 1;
    cur_pos  = keyframes_[latest].pose.t_;
    cur_time = keyframes_[latest].timestamp;
  }

  // --- Step 2: check dedup guard (separate lock — no nesting) ---
  {
    std::lock_guard<std::mutex> idx_lock(mtx_loop_index_);
    if (loop_index_container_.count(latest)) return false;
  }

  // --- Step 3: build PCL position cloud and KD-tree for radius search ---
  pcl::PointCloud<pcl::PointXYZ>::Ptr pose_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  std::vector<double> timestamps_copy;

  {
    std::lock_guard<std::mutex> lock(mtx_keyframes_);
    pose_cloud->reserve(keyframes_.size());
    timestamps_copy.reserve(keyframes_.size());
    for (const auto& kf : keyframes_) {
      pcl::PointXYZ p;
      p.x = kf.pose.t_[0];
      p.y = kf.pose.t_[1];
      p.z = kf.pose.t_[2];
      pose_cloud->push_back(p);
      timestamps_copy.push_back(kf.timestamp);
    }
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(pose_cloud);

  std::vector<int>   indices;
  std::vector<float> sq_dists;
  pcl::PointXYZ query;
  query.x = cur_pos[0];
  query.y = cur_pos[1];
  query.z = cur_pos[2];

  kdtree.radiusSearch(query,
      static_cast<float>(g_history_keyframe_search_radius),
      indices, sq_dists);

  // Take the closest result that is far enough in time.
  int best_pre = -1;
  for (int idx : indices) {
    if (idx >= latest) continue;
    if ((cur_time - timestamps_copy[idx]) < g_history_keyframe_search_time_diff) continue;
    best_pre = idx;
    break;   // sorted by distance — first valid is closest
  }

  if (best_pre < 0) return false;

  *cur_id = latest;
  *pre_id = best_pre;
  return true;
}


// ============================================================================
// Submap builder — caller MUST hold mtx_keyframes_
// ============================================================================

BASIC::CloudPtr SuperLIOLoop::buildSubmap(int center_id, int half_window)
{
  CloudPtr submap(new PointCloudType());

  const int n = static_cast<int>(keyframes_.size());
  for (int i = center_id - half_window; i <= center_id + half_window; ++i) {
    if (i < 0 || i >= n) continue;
    const auto& kf = keyframes_[i];
    for (const auto& pt : *kf.cloud) {
      const V3 pw = kf.pose * V3(pt.x, pt.y, pt.z);
      PointType p;
      p.x = pw[0]; p.y = pw[1]; p.z = pw[2];
      p.intensity = pt.intensity;
      submap->push_back(p);
    }
  }

  // Downsample the submap with the configured density voxel size.
  if (!submap->empty()) {
    pcl::VoxelGrid<PointType> vf;
    const float leaf = static_cast<float>(g_surrounding_keyframe_density);
    vf.setLeafSize(leaf, leaf, leaf);
    vf.setInputCloud(submap);
    CloudPtr filtered(new PointCloudType());
    vf.filter(*filtered);
    return filtered;
  }

  return submap;
}


// ============================================================================
// Main loop closure work — runs in background thread
// ============================================================================

void SuperLIOLoop::performLoopClosure()
{
  int cur_id = -1, pre_id = -1;
  if (!detectLoopClosureDistance(&cur_id, &pre_id)) return;

  // Extract the current keyframe cloud (single frame, world frame) and the
  // historical submap while holding the keyframe lock.
  CloudPtr cur_cloud_world(new PointCloudType());
  CloudPtr pre_cloud_world;
  SE3      cur_pose_wrong;

  {
    std::lock_guard<std::mutex> lock(mtx_keyframes_);

    const auto& kf_cur = keyframes_[cur_id];
    cur_pose_wrong = kf_cur.pose;

    // Transform current keyframe cloud to world frame.
    for (const auto& pt : *kf_cur.cloud) {
      const V3 pw = kf_cur.pose * V3(pt.x, pt.y, pt.z);
      PointType p;
      p.x = pw[0]; p.y = pw[1]; p.z = pw[2];
      p.intensity = pt.intensity;
      cur_cloud_world->push_back(p);
    }

    // Build historical submap around pre_id.
    pre_cloud_world = buildSubmap(pre_id, g_history_keyframe_search_num);
  }

  // Sanity size check (mirrors LIO-SAM thresholds).
  if (cur_cloud_world->size() < 300 || pre_cloud_world->size() < 1000) return;

  // Optional: publish history cloud for RViz inspection.
  if (pub_history_cloud_ && pub_history_cloud_->get_subscription_count() > 0) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*pre_cloud_world, msg);
    msg.header.frame_id = "world";
    pub_history_cloud_->publish(msg);
  }

  // ------------------------------------------------------------------
  // ICP alignment (source = current frame, target = historical submap)
  // ------------------------------------------------------------------
  static pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(g_history_keyframe_search_radius * 2.0);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  icp.setInputSource(cur_cloud_world);
  icp.setInputTarget(pre_cloud_world);
  CloudPtr icp_result(new PointCloudType());
  icp.align(*icp_result);

  if (!icp.hasConverged() ||
       icp.getFitnessScore() > g_history_keyframe_fitness_score) {
    LOG(INFO) << YELLOW << " ---> [LoopClosure]: ICP rejected. score="
              << icp.getFitnessScore()
              << " (threshold=" << g_history_keyframe_fitness_score << ")" << RESET;
    return;
  }

  LOG(INFO) << GREEN << " ---> [LoopClosure]: loop confirmed! "
            << "cur=" << cur_id << "  ref=" << pre_id
            << "  score=" << icp.getFitnessScore() << RESET;

  // Optional: publish aligned ICP result for RViz.
  if (pub_icp_cloud_ && pub_icp_cloud_->get_subscription_count() > 0) {
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*icp_result, msg);
    msg.header.frame_id = "world";
    pub_icp_cloud_->publish(msg);
  }

  // ------------------------------------------------------------------
  // Compute corrected world pose of keyframes_[cur_id]
  //
  //   T_correction  = icp.getFinalTransformation()    (world → world)
  //   T_wrong       = current (drifted) pose in world
  //   T_correct     = T_correction * T_wrong
  // ------------------------------------------------------------------
  const Eigen::Matrix4f T_corr_4f = icp.getFinalTransformation();

  Eigen::Matrix4f T_wrong_4f = Eigen::Matrix4f::Identity();
  T_wrong_4f.block<3, 3>(0, 0) = cur_pose_wrong.R_.cast<float>();
  T_wrong_4f.block<3, 1>(0, 3) = cur_pose_wrong.t_.cast<float>();

  const Eigen::Matrix4f T_correct_4f = T_corr_4f * T_wrong_4f;

  const M3 R_correct = T_correct_4f.block<3, 3>(0, 0).cast<scalar>();
  const V3 t_correct = T_correct_4f.block<3, 1>(0, 3).cast<scalar>();
  const SE3 corrected_pose(SO3(R_correct), t_correct);

  // Push result to main thread.
  {
    std::lock_guard<std::mutex> lock(mtx_loop_);
    loop_queue_.push(LoopResult{cur_id, pre_id, corrected_pose});
  }

  // Register this pair so we don't re-attempt it.
  {
    std::lock_guard<std::mutex> idx_lock(mtx_loop_index_);
    loop_index_container_[cur_id] = pre_id;
  }
}


// ============================================================================
// Apply loop correction — called in the main thread at the top of stateProcess
// ============================================================================

void SuperLIOLoop::applyLoopCorrection()
{
  LoopResult result;
  {
    std::lock_guard<std::mutex> lock(mtx_loop_);
    if (loop_queue_.empty()) return;
    result = loop_queue_.front();
    loop_queue_.pop();
  }

  // Retrieve the (wrong) pose of the loop keyframe.
  SE3 T_wrong;
  {
    std::lock_guard<std::mutex> lock(mtx_keyframes_);
    T_wrong = keyframes_[result.cur_id].pose;
  }

  // delta = T_correct * T_wrong^{-1}
  // Applied to any world-frame quantity q: q_new = delta * q
  const M3 R_wi    = T_wrong.R_.transpose();          // R_wrong^{-1}
  const M3 delta_R = result.corrected_pose.R_ * R_wi;
  const V3 delta_t = result.corrected_pose.t_ - delta_R * T_wrong.t_;

  const float trace_dr = delta_R.trace();
  const float angle_dr = std::acos(std::clamp((trace_dr - 1.0f) / 2.0f, -1.0f, 1.0f));

  LOG(INFO) << GREEN << " ---> [LoopClosure]: applying correction. "
            << "dt=" << delta_t.norm() << " m  "
            << "dangle=" << angle_dr * 180.f / static_cast<float>(M_PI) << " deg"
            << RESET;

  // --- Correct ESKF state (main thread owns kf_ — safe, no extra lock) ---
  {
    SysState state = kf_->GetSysState();
    state.R = SO3(delta_R * state.R.R_);
    state.p = delta_R * state.p + delta_t;
    state.v = delta_R * state.v;          // rotate velocity into corrected frame
    kf_->SetX(state);
    last_pose_ = kf_->GetSE3();
  }

  // --- Correct all keyframe poses at and after cur_id ---
  {
    std::lock_guard<std::mutex> lock(mtx_keyframes_);
    for (auto& kf : keyframes_) {
      if (kf.id < result.cur_id) continue;
      kf.pose = SE3(SO3(delta_R * kf.pose.R_),
                    delta_R * kf.pose.t_ + delta_t);
    }
  }

  // --- Rebuild the voxel map from corrected keyframes ---
  rebuildMap();
}


// ============================================================================
// Voxel map rebuild
// ============================================================================

void SuperLIOLoop::rebuildMap()
{
  // Reset the incremental voxel map — same options as in SuperLIO::init().
  ivox_.reset(new OctVoxMapType(
      OctVoxMapType::Options{g_ivox_resolution, g_ivox_capacity}));

  VV3 pts;
  {
    std::lock_guard<std::mutex> lock(mtx_keyframes_);

    LOG(INFO) << YELLOW << " ---> [LoopClosure]: rebuilding map from "
              << keyframes_.size() << " keyframes..." << RESET;

    for (const auto& kf : keyframes_) {
      pts.clear();
      pts.reserve(kf.cloud->size());
      for (const auto& pt : *kf.cloud) {
        pts.push_back(kf.pose * V3(pt.x, pt.y, pt.z));
      }
      ivox_->insert(pts);
    }
  }

  LOG(INFO) << GREEN << " ---> [LoopClosure]: map rebuild complete." << RESET;
}


// ============================================================================
// RViz visualisation — called from the background thread
// ============================================================================

void SuperLIOLoop::visualizeLoopClosure()
{
  if (!pub_loop_markers_) return;

  // Copy the loop index map (mtx_loop_index_) — no nesting with mtx_keyframes_.
  std::map<int, int> loop_copy;
  {
    std::lock_guard<std::mutex> idx_lock(mtx_loop_index_);
    if (loop_index_container_.empty()) return;
    loop_copy = loop_index_container_;
  }

  visualization_msgs::msg::MarkerArray marker_array;

  // --- Nodes (sphere list) ---
  visualization_msgs::msg::Marker nodes;
  nodes.header.frame_id = "world";
  nodes.action = visualization_msgs::msg::Marker::ADD;
  nodes.type   = visualization_msgs::msg::Marker::SPHERE_LIST;
  nodes.ns     = "loop_nodes";
  nodes.id     = 0;
  nodes.pose.orientation.w = 1.0;
  nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.3;
  nodes.color.r = 0.0f; nodes.color.g = 0.8f;
  nodes.color.b = 1.0f; nodes.color.a = 1.0f;

  // --- Edges (line list) ---
  visualization_msgs::msg::Marker edges;
  edges.header.frame_id = "world";
  edges.action = visualization_msgs::msg::Marker::ADD;
  edges.type   = visualization_msgs::msg::Marker::LINE_LIST;
  edges.ns     = "loop_edges";
  edges.id     = 1;
  edges.pose.orientation.w = 1.0;
  edges.scale.x = 0.1;
  edges.color.r = 0.9f; edges.color.g = 0.9f;
  edges.color.b = 0.0f; edges.color.a = 1.0f;

  {
    std::lock_guard<std::mutex> kf_lock(mtx_keyframes_);
    const int n = static_cast<int>(keyframes_.size());

    for (const auto& [cur, ref] : loop_copy) {
      if (cur >= n || ref >= n) continue;

      const V3& p_cur = keyframes_[cur].pose.t_;
      const V3& p_ref = keyframes_[ref].pose.t_;

      geometry_msgs::msg::Point pt_cur, pt_ref;
      pt_cur.x = p_cur[0]; pt_cur.y = p_cur[1]; pt_cur.z = p_cur[2];
      pt_ref.x = p_ref[0]; pt_ref.y = p_ref[1]; pt_ref.z = p_ref[2];

      nodes.points.push_back(pt_cur);
      nodes.points.push_back(pt_ref);
      edges.points.push_back(pt_cur);
      edges.points.push_back(pt_ref);
    }
  }

  marker_array.markers.push_back(std::move(nodes));
  marker_array.markers.push_back(std::move(edges));
  pub_loop_markers_->publish(marker_array);
}

} // namespace LI2Sup
