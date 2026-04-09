
#include "lio/super_lio_loop.h"
#include "basic/logs.h"
#include <iomanip>

using namespace BASIC;

namespace LI2Sup {

// ============================================================================
//  Constructor / Destructor
// ============================================================================

SuperLIOLoop::SuperLIOLoop()
{
  kf_positions_.reset(new pcl::PointCloud<pcl::PointXYZ>());

  // ── Lidar rotation matrix (Rz * Ry * Rx) ─────────────────────────────────
  sensor_rotation_x_ = static_cast<float>(g_lidar_rotation_x);
  sensor_rotation_y_ = static_cast<float>(g_lidar_rotation_y);
  sensor_rotation_z_ = static_cast<float>(g_lidar_rotation_z);

  Eigen::Matrix3f R =
    (Eigen::AngleAxisf(sensor_rotation_z_, Eigen::Vector3f::UnitZ()) *
     Eigen::AngleAxisf(sensor_rotation_y_, Eigen::Vector3f::UnitY()) *
     Eigen::AngleAxisf(sensor_rotation_x_, Eigen::Vector3f::UnitX()))
    .toRotationMatrix();

  rotation_matrix_                   = Eigen::Matrix4f::Identity();
  rotation_matrix_.block<3, 3>(0, 0) = R;

  // STD config — filled from yaml-driven globals (see livox_360.yaml lio.std.*)
  std_cfg_.ds_size_                    = g_std_ds_size;
  std_cfg_.maximum_corner_num_         = g_std_maximum_corner_num;
  std_cfg_.plane_merge_normal_thre_    = g_std_plane_merge_normal_thre;
  std_cfg_.plane_merge_dis_thre_       = g_std_plane_merge_dis_thre;
  std_cfg_.plane_detection_thre_       = g_std_plane_detection_thre;
  std_cfg_.voxel_size_                 = g_std_voxel_size;
  std_cfg_.voxel_init_num_             = g_std_voxel_init_num;
  std_cfg_.proj_image_resolution_      = g_std_proj_image_resolution;
  std_cfg_.proj_dis_min_               = g_std_proj_dis_min;
  std_cfg_.proj_dis_max_               = g_std_proj_dis_max;
  std_cfg_.corner_thre_                = g_std_corner_thre;
  std_cfg_.descriptor_near_num_        = g_std_descriptor_near_num;
  std_cfg_.descriptor_min_len_         = g_std_descriptor_min_len;
  std_cfg_.descriptor_max_len_         = g_std_descriptor_max_len;
  std_cfg_.non_max_suppression_radius_ = g_std_non_max_suppression_radius;
  std_cfg_.std_side_resolution_        = g_std_std_side_resolution;
  std_cfg_.candidate_num_              = g_std_candidate_num;
  std_cfg_.rough_dis_threshold_        = g_std_rough_dis_threshold;
  std_cfg_.vertex_diff_threshold_      = g_std_vertex_diff_threshold;
  std_cfg_.icp_threshold_              = g_std_icp_threshold;
  std_cfg_.normal_threshold_           = g_std_normal_threshold;
  std_cfg_.dis_threshold_              = g_std_dis_threshold;
  std_cfg_.sub_frame_num_              = g_std_sub_frame_num;
  std_cfg_.skip_near_num_             = g_std_skip_near_num;
  std_manager_ = STDescManager(std_cfg_);
}

// ============================================================================
//  DownSample()  – rotate scan in place, then delegate to base downsampler
// ============================================================================

void SuperLIOLoop::DownSample()
{
  if (scan_undistort_full_ && !scan_undistort_full_->empty()) {
    pcl::transformPointCloud(*scan_undistort_full_, *scan_undistort_full_, rotation_matrix_);
  }
  SuperLIO::DownSample();
}

SuperLIOLoop::~SuperLIOLoop()
{
  // Signal background threads to stop and wait for them to finish.
  loop_thread_stop_.store(true);
  if (loop_thread_.joinable()) loop_thread_.join();

  if (g_save_map) {
    std_thread_stop_.store(true);
    if (std_thread_.joinable()) std_thread_.join();
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

  LOG(INFO) << GREEN << " ---> [SuperLIOLoop]: Path publisher active on ----------> /super_lio/corrected_path" << RESET;

  // ── STD descriptor thread — only started when save_map is true so that
  //    descriptor extraction (expensive) is skipped during odometry-only runs.
  if (g_save_map) {
    std_thread_ = std::thread(&SuperLIOLoop::stdDescriptorThread, this);
    LOG(INFO) << GREEN << " ---> [SuperLIOLoop]: STD descriptor thread started." << RESET;
  } else {
    LOG(INFO) << YELLOW << " ---> [SuperLIOLoop]: STD descriptor thread SKIPPED (save_map=false)." << RESET;
  }

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

  if (!g_loop_closure_enable && !g_local_map_enable) return;

  if (new_keyframe_) {
    new_keyframe_ = false;

    LOG(INFO) << YELLOW << " ---> [Output]: new_keyframe kf_count_=" << kf_count_
              << " flag_rebuild=" << flag_rebuild_.load()
              << " prune_counter=" << prune_counter_
              << RESET;

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

  // Build the accumulated world cloud inside the same mtx_kf_ scope so we
  // only acquire the lock once instead of twice.
  pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud;

  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    keyframes_.push_back(kf);

    pcl::PointXYZ kf_pos;
    kf_pos.x = static_cast<float>(last_pose_.t_(0));
    kf_pos.y = static_cast<float>(last_pose_.t_(1));
    kf_pos.z = static_cast<float>(last_pose_.t_(2));
    kf_positions_->push_back(kf_pos);

    // Build accumulated world cloud for STD extraction (background thread).
    // Only when save_map is true; otherwise STD extraction is skipped.
    if (g_save_map) {
      world_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
      int n     = static_cast<int>(keyframes_.size());
      int start = std::max(0, n - std_cfg_.sub_frame_num_);
      for (int i = start; i < n; ++i) {
        const auto &kfi = keyframes_[i];
        for (const auto &pt : kfi.cloud->points) {
          V3 pw = kfi.pose * V3(pt.x, pt.y, pt.z);
          pcl::PointXYZI p;
          p.x = pw.x(); p.y = pw.y(); p.z = pw.z();
          p.intensity = pt.intensity;
          world_cloud->push_back(p);
        }
      }
    }
  }

  if (world_cloud) {
    std::lock_guard<std::mutex> qlock(mtx_std_queue_);
    std_cloud_queue_.push(world_cloud);
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

  LOG(INFO) << YELLOW << " ---> [Graph]: OdomFactor added kf_count_=" << kf_count_
            << " last_kf_se3_.t="
            << last_kf_se3_.t_(0) << ","
            << last_kf_se3_.t_(1) << ","
            << last_kf_se3_.t_(2)
            << RESET;
}

void SuperLIOLoop::addLoopFactor()
{
  std::lock_guard<std::mutex> lock(mtx_loop_);
  if (loop_index_queue_.empty()) return;

  LOG(INFO) << YELLOW << " ---> [Graph]: LoopFactors consumed count="
            << loop_index_queue_.size()
            << RESET;

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
  LOG(INFO) << YELLOW << " ---> [Graph]: iSAM2 update loop_closed=" << loop_closed
            << " graph_factors=" << gtsam_graph_.size()
            << " init_vals=" << gtsam_init_vals_.size()
            << " estimate_size=" << isam_estimate_.size()
            << RESET;

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

  LOG(INFO) << YELLOW << " ---> [Rebuild]: START kf_count_=" << kf_count_
            << " keyframes.size()=" << keyframes_.size()
            << " corrected.size()=" << corrected.size()
            << RESET;

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

  LOG(INFO) << GREEN << " ---> [Rebuild]: DONE ivox reinserted kf_count_=" << kf_count_
            << " keyframes.size()=" << keyframes_.size()
            << RESET;
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

  {
    size_t pending = 0;
    {
      std::lock_guard<std::mutex> lk(mtx_loop_);
      pending = loop_index_queue_.size();
    }
    LOG(INFO) << YELLOW << " ---> [Loop]: Candidate detected cur=" << cur_id
              << " pre=" << pre_id
              << " kf_count_=" << kf_count_
              << " flag_rebuild=" << flag_rebuild_.load()
              << " pending_loops=" << pending
              << RESET;
  }

  SE3 cur_pose, pre_pose;
  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    cur_pose = keyframes_[cur_id].pose;
    pre_pose = keyframes_[pre_id].pose;
  }

  // ── Try STD plane ICP first — better Z accuracy than point ICP ───────────
  if (g_save_map) {
    bool stds_ready = false;
    std::vector<STDesc> cur_stds;
    {
      std::shared_lock<std::shared_mutex> lock(mtx_std_manager_);
      stds_ready = (cur_id < (int)kf_stds_.size() &&
                    pre_id < (int)kf_stds_.size() &&
                    !kf_stds_[cur_id].empty());
      if (stds_ready) cur_stds = kf_stds_[cur_id];
    }

    if (!stds_ready) {
      LOG(INFO) << YELLOW << " ---> [Loop+STD]: STD thread lagging "
                << "(kf_stds_.size()=" << kf_stds_.size()
                << " cur_id=" << cur_id << "), falling back to ICP." << RESET;
    }

    if (stds_ready) {
      std::pair<int, double>                           loop_result;
      std::pair<Eigen::Vector3d, Eigen::Matrix3d>      loop_transform;
      std::vector<std::pair<STDesc, STDesc>>           loop_pairs;
      bool plane_cloud_available = false;

      {
        // shared_lock: SearchLoop is a read-only operation;
        // concurrent reads are fine and the STD write thread won't be starved.
        std::shared_lock<std::shared_mutex> lock(mtx_std_manager_);
        plane_cloud_available =
            (cur_id < (int)std_manager_.plane_cloud_vec_.size() &&
             pre_id < (int)std_manager_.plane_cloud_vec_.size());

        if (plane_cloud_available) {
          // skip_geometric_verify=true: plane_geometric_verify uses dis_threshold_=0.3m
          // but Z drift can be 0.3-0.9m, so ALL plane correspondences fail → score≈0.
          // Instead score = min(1.0, max_vote/20.0) based purely on triangle consistency.
          std_manager_.SearchLoop(cur_stds,
                                  std_manager_.plane_cloud_vec_[cur_id],
                                  loop_result, loop_transform, loop_pairs,
                                  /*skip_geometric_verify=*/true);

          LOG(INFO) << YELLOW << " ---> [STD Diag]"
                    << " cur_id=" << cur_id
                    << " pre_id=" << pre_id
                    << " cur_stds=" << cur_stds.size()
                    << " db_size=" << std_manager_.data_base_.size()
                    << " plane_vec_size=" << std_manager_.plane_cloud_vec_.size()
                    << " cur_frame_id=" << (cur_stds.empty() ? -1 : (int)cur_stds[0].frame_id_)
                    << " loop_result.first=" << loop_result.first
                    << " loop_result.second=" << loop_result.second
                    << RESET;

          // Do NOT call PlaneGeomrtricIcp here: it uses dis_threshold_=0.3m for plane
          // correspondences, but the triangle solver's Z component is corrupted by
          // ESKF Z drift — p2plane >> 0.3 → zero residuals → transform unchanged or
          // diverges. Z correction is done in the acceptance block below.
        }
      }

      if (plane_cloud_available &&
          loop_result.first >= 0 &&
          loop_result.second > std_cfg_.icp_threshold_) {

        // Use the STD-matched frame as the actual loop anchor.
        // detectLoop's pre_id is a spatial proximity heuristic; STD finds the
        // geometrically consistent match — they may differ by several frames.
        int std_pre_id = loop_result.first;
        SE3 std_pre_pose;
        double std_pre_stamp = 0.0;
        double cur_stamp     = 0.0;
        {
          std::lock_guard<std::mutex> lk(mtx_kf_);
          if (std_pre_id >= (int)keyframes_.size()) goto fallback_icp;
          std_pre_pose  = keyframes_[std_pre_id].pose;
          std_pre_stamp = keyframes_[std_pre_id].timestamp;
          cur_stamp     = keyframes_[cur_id].timestamp;
        }

        // ── Spatial sanity check ────────────────────────────────────────────
        // STD with skip_geometric_verify can match triangles from structurally
        // similar but physically distant locations.  We validate the STD match
        // against BOTH reference frames:
        //  (a) std_pre must be within g_loop_search_radius of cur in XY
        //  (b) std_pre must be within g_loop_search_radius of pre (KD-tree candidate) in XY
        // Using XY only because Z is drifted.
        {
          float xy_dist_cur = (std_pre_pose.t_.head<2>() -
                               cur_pose.t_.head<2>()).norm();
          float xy_dist_pre = (std_pre_pose.t_.head<2>() -
                               pre_pose.t_.head<2>()).norm();
          float limit = static_cast<float>(g_loop_search_radius);
          if (xy_dist_cur > limit || xy_dist_pre > limit) {
            LOG(INFO) << YELLOW << " ---> [Loop+STD]: Rejected — std_pre XY too far"
                      << " std_pre=" << std_pre_id
                      << " xy_from_cur=" << std::fixed << std::setprecision(2) << xy_dist_cur
                      << "m xy_from_pre=" << xy_dist_pre
                      << "m > radius=" << g_loop_search_radius << "m" << RESET;
            goto fallback_icp;
          }
        }

        // ── Temporal sanity check ───────────────────────────────────────────
        // std_pre must be old enough to be from a different traversal pass.
        {
          double time_gap = cur_stamp - std_pre_stamp;
          if (time_gap < g_loop_search_time_diff) {
            LOG(INFO) << YELLOW << " ---> [Loop+STD]: Rejected — std_pre too recent"
                      << " std_pre=" << std_pre_id
                      << " time_gap=" << std::fixed << std::setprecision(1) << time_gap
                      << "s < " << g_loop_search_time_diff << "s" << RESET;
            goto fallback_icp;
          }
        }

        // ── Z displacement sanity check ─────────────────────────────────────
        // Pose Z difference > 5m almost certainly means a bad STD match or
        // extreme accumulated drift — reject rather than apply a huge Z jump.
        // Must be checked before any goto that crosses non-trivial constructors.
        {
          double dz = (double)std_pre_pose.t_(2) - (double)cur_pose.t_(2);
          if (std::abs(dz) > 5.0) {
            LOG(INFO) << YELLOW << " ---> [Loop+STD]: Rejected — dZ too large"
                      << " std_pre=" << std_pre_id
                      << " dZ=" << std::fixed << std::setprecision(3) << dz << "m" << RESET;
            goto fallback_icp;
          }
        }

        {
          // Z-only correction strategy:
          // ESKF XY odometry is accurate — only Z drifts from IMU bias.
          // Applying a full 6DOF STD transform to pose_from_corrected can cause
          // large XY "teleportation" (seen: 15m jump in logs) because the triangle
          // solver's XY also carries noise when matched frames are far apart.
          //
          // Instead: keep cur_pose's rotation and XY, correct only Z to match
          // std_pre_pose's Z level.  GTSAM noise model has very large variance on
          // XY and rotation → those DOFs are left to odometry; only Z is tightened.
          double dz = (double)std_pre_pose.t_(2) - (double)cur_pose.t_(2);

          gtsam::Pose3 pose_from_corrected(
              gtsam::Rot3(cur_pose.R_.cast<double>()),            // keep ESKF rotation
              gtsam::Point3(cur_pose.t_(0), cur_pose.t_(1),       // keep ESKF XY
                            (double)std_pre_pose.t_(2)));          // correct Z only

          gtsam::Pose3 pose_to = toGtsamPose(std_pre_pose);

          // Noise ordering: [rx, ry, rz, tx, ty, tz].
          // Large XY/rotation variance → those DOFs unconstrained by this factor.
          // Tight Z variance → corrects accumulated IMU Z bias.
          gtsam::Vector6 noise_vec;
          noise_vec << 100.0, 100.0, 100.0,  // rx, ry, rz — essentially free
                       100.0, 100.0, 0.04;   // tx, ty — free; tz σ=0.2m
          auto noise = gtsam::noiseModel::Diagonal::Variances(noise_vec);

          {
            std::lock_guard<std::mutex> lock(mtx_loop_);
            loop_index_queue_.emplace_back(std_pre_id, cur_id);
            loop_pose_queue_.push_back(pose_to.between(pose_from_corrected));
            loop_noise_queue_.push_back(noise);
            loop_history_[cur_id] = std_pre_id;
          }

          flag_rebuild_.store(true);
          LOG(INFO) << GREEN << " ---> [Loop+STD]: Accepted (Z-only)! cur=" << cur_id
                    << " std_pre=" << std_pre_id
                    << " score=" << std::fixed << std::setprecision(3) << loop_result.second
                    << " dZ=" << std::fixed << std::setprecision(3) << dz << "m" << RESET;
          return true;
        }
      }
      fallback_icp:;
    }
  }

  // ── Fallback: vanilla point ICP ───────────────────────────────────────────
  CloudPtr cur_cloud(new PointCloudType());
  CloudPtr pre_cloud(new PointCloudType());

  buildLocalMap(cur_cloud, cur_id, 0);                      // current: single keyframe
  buildLocalMap(pre_cloud, pre_id, g_loop_search_num);      // previous: wide neighbourhood

  LOG(INFO) << YELLOW << " ---> [Loop]: Local maps built"
            << " cur_cloud=" << cur_cloud->size()
            << " pre_cloud=" << pre_cloud->size()
            << " flag_rebuild=" << flag_rebuild_.load()
            << RESET;

  if (cur_cloud->size() < 300 || pre_cloud->size() < 1000) {
    LOG(WARNING) << YELLOW << " ---> [Loop]: Candidate clouds too small ("
                 << cur_cloud->size() << " / " << pre_cloud->size() << "), skipped." << RESET;
    return false;
  }

  // ── Pose-based Z correction for ICP initial guess ────────────────────────
  // The ESKF accumulates Z drift (IMU bias), so cur_pose.t_(2) ≠ pre_pose.t_(2)
  // even though both frames visited the same physical Z height.
  // pre_pose is from early in the trajectory (little drift); cur_pose is drifted.
  // Applying dz = pre_pose.t_(2) - cur_pose.t_(2) as the ICP initial guess
  // shifts cur_cloud to the correct Z level before ICP runs.
  //
  // STD triangle transforms were tried but produce degenerate Z values (tens of
  // metres) because the plane corners themselves are at drifted heights — the
  // solver finds a transform that is "consistent" in 3D but carries a huge Z
  // component. The pose difference is simpler and directly correct.
  Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
  {
    float dz = (float)(pre_pose.t_(2) - cur_pose.t_(2));
    if (std::abs(dz) > 0.05f && std::abs(dz) < 5.0f) {
      initial_guess(2, 3) = dz;
      LOG(INFO) << YELLOW << " ---> [Loop]: Pose Z correction for ICP"
                << " cur_z=" << std::fixed << std::setprecision(3) << cur_pose.t_(2)
                << " pre_z=" << pre_pose.t_(2)
                << " dz=" << dz << "m" << RESET;
    }
  }

  // ── ICP alignment (GICP) ─────────────────────────────────────────────────
  // GICP (Generalized ICP) estimates local covariance structure around each
  // point, making it far more robust to Z drift than vanilla point-to-point
  // ICP — flat floors no longer cause Z to "slide" unconstrained.
  pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(3.0);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);

  icp.setInputSource(cur_cloud);
  icp.setInputTarget(pre_cloud);
  CloudPtr unused(new PointCloudType());
  icp.align(*unused, initial_guess);

  if (!icp.hasConverged() || icp.getFitnessScore() > g_loop_fitness_score) {
    LOG(INFO) << YELLOW << " ---> [Loop]: ICP rejected (converged="
              << icp.hasConverged()
              << ", score=" << icp.getFitnessScore() << ")." << RESET;
    std::lock_guard<std::mutex> lock(mtx_loop_);
    if (++loop_failed_attempts_[cur_id] >= 3) {
      loop_history_[cur_id] = -1;   // -1 = exhausted; detectLoop skips it
      LOG(WARNING) << YELLOW << " ---> [Loop]: Giving up on cur=" << cur_id
                   << " after 3 failed attempts." << RESET;
    }
    return false;
  }

  LOG(INFO) << GREEN << " ---> [Loop]: Accepted via ICP! cur=" << cur_id
            << " pre=" << pre_id
            << " score=" << icp.getFitnessScore() << RESET;

  // ── Compute pose correction ───────────────────────────────────────────────
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
  // BetweenFactor(tar=pre, src=cur, Z) where Z = T_pre^{-1} * T_cur_corrected
  // = pose_to.between(pose_from) — relative transform in pre's local frame.
  {
    std::lock_guard<std::mutex> lock(mtx_loop_);
    loop_index_queue_.emplace_back(pre_id, cur_id);   // tar=pre first, src=cur second
    loop_pose_queue_.push_back(pose_to.between(pose_from));
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
    if (keyframes_.size() < 3) return false;   // need at least n_kf-2 >= 1
    *positions_copy = *kf_positions_;
    n_kf            = static_cast<int>(keyframes_.size());
    latest_stamp    = keyframes_.back().timestamp;
  }

  // Use n_kf-2 (second-most-recent) so the STD thread has always processed
  // this keyframe by the time the loop thread inspects it. The off-by-one
  // (kf_stds_.size()==cur_id) is eliminated at the cost of 0.5m latency.
  int loop_key_cur = n_kf - 2;
  if (loop_key_cur < 1) return false;

  // Avoid re-detecting a loop we already found (or exhausted) for this keyframe.
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

  // Use cur_id's own timestamp for the gap check — not back()'s stamp,
  // which is 1 frame newer and would slightly overestimate the time gap.
  double cur_stamp = 0.0;
  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    cur_stamp = keyframes_[loop_key_cur].timestamp;
  }

  int loop_key_pre = -1;
  {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    for (int idx : indices) {
      double stamp_diff = cur_stamp - keyframes_[idx].timestamp;
      if (stamp_diff > g_loop_search_time_diff) {
        loop_key_pre = idx;
        break;
      }
    }
  }

  double selected_stamp_diff = -1.0;
  if (loop_key_pre >= 0) {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    selected_stamp_diff = cur_stamp - keyframes_[loop_key_pre].timestamp;
  }

  LOG(INFO) << YELLOW << " ---> [Detect]: cur=" << loop_key_cur
            << " radius_candidates=" << indices.size()
            << " selected_pre=" << loop_key_pre
            << " stamp_diff=" << std::fixed << std::setprecision(1) << selected_stamp_diff
            << RESET;

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

  // Snapshot loop_history_ under mtx_loop_ FIRST — never hold both locks at once.
  // (Acquiring mtx_kf_ while holding mtx_loop_ would invert the lock order used
  // elsewhere and create a potential deadlock with performLoopClosure().)
  std::unordered_map<int, int> history_snap;
  {
    std::lock_guard<std::mutex> lk(mtx_loop_);
    history_snap = loop_history_;
  }

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

  for (const auto& [from, to] : history_snap) {
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

// ============================================================================
//  STD descriptor background thread
//
//  Drains std_cloud_queue_, runs the expensive GenerateSTDescs + AddSTDescs
//  pipeline off the main LIO thread.  The queue decouples keyframe insertion
//  (fast, main thread) from descriptor extraction (slow, this thread).
// ============================================================================

void SuperLIOLoop::stdDescriptorThread()
{
  while (true) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;

    {
      std::lock_guard<std::mutex> lock(mtx_std_queue_);
      if (!std_cloud_queue_.empty()) {
        cloud = std_cloud_queue_.front();
        std_cloud_queue_.pop();
      }
    }

    if (!cloud) {
      {
        std::lock_guard<std::mutex> lock(mtx_std_queue_);
        size_t depth = std_cloud_queue_.size();
        if (depth > 5) {
          LOG(WARNING) << YELLOW << " ---> [STD]: Queue backing up depth="
                       << depth << RESET;
        }
      }
      // Only exit after the queue is fully drained — not just when the stop
      // flag is set, so we don't silently drop queued clouds on shutdown.
      if (std_thread_stop_.load()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    down_sampling_voxel(*cloud, std_cfg_.ds_size_);

    std::vector<STDesc> stds;
    {
      std::unique_lock<std::shared_mutex> lock(mtx_std_manager_);
      std_manager_.GenerateSTDescs(cloud, stds);

      LOG(INFO) << YELLOW << " ---> [STD DEBUG] planes="
                << (std_manager_.plane_cloud_vec_.empty() ? 0
                    : std_manager_.plane_cloud_vec_.back()->size())
                << " corners="
                << (std_manager_.corner_cloud_vec_.empty() ? 0
                    : std_manager_.corner_cloud_vec_.back()->size())
                << " descs=" << stds.size() << RESET;

      std_manager_.AddSTDescs(stds);
      kf_stds_.push_back(stds);

      // Diagnostic: verify frame_id_ vs plane_cloud_vec_ alignment.
      // frame_id_ in descriptors = plane_cloud_vec_ index (both 0-based).
      // After AddSTDescs, current_frame_id_ = plane_cloud_vec_.size().
      // expected_offset should always be 0 — if it's 1, there IS an off-by-one.
      LOG(INFO) << YELLOW << " ---> [STD] frame_id offset check:"
                << " current_frame_id_=" << std_manager_.current_frame_id_
                << " plane_cloud_vec_.size()=" << std_manager_.plane_cloud_vec_.size()
                << " kf_stds_.size()=" << kf_stds_.size()
                << " expected_offset=" << (std_manager_.current_frame_id_
                                           - (int)std_manager_.plane_cloud_vec_.size())
                << RESET;
    }

    LOG(INFO) << GREEN << " ---> [STD] cloud_size=" << cloud->size()
              << " descs=" << stds.size() << RESET;
  }
}

// ============================================================================
//  saveSTDDatabase()
// ============================================================================

void SuperLIOLoop::saveSTDDatabase()
{
  // Stop the STD thread and join it — this is the only safe barrier.
  // The thread drains the queue internally before exiting (loop checks
  // std_thread_stop_ after each cloud), so no extra sleep is needed.
  // The destructor's joinable() guard prevents a double-join.
  LOG(INFO) << YELLOW << " ---> [STD] Stopping descriptor thread before save..." << RESET;
  std_thread_stop_.store(true);
  if (std_thread_.joinable()) std_thread_.join();

  std::string db_dir = g_save_map_dir + "/std_db";
  LOG(INFO) << YELLOW << " ---> [STD] Saving database to: " << db_dir << RESET;
  if (std_manager_.Save(db_dir)) {
    LOG(INFO) << GREEN << " ---> [STD] Save OK."
              << "  frames=" << std_manager_.plane_cloud_vec_.size() << RESET;
  } else {
    LOG(ERROR) << RED << " ---> [STD] Save FAILED." << RESET;
  }

  // Also save keyframe poses for visualization
  std::string poses_path = g_save_map_dir + "/std_db/keyframe_poses.bin";
  std::ofstream f(poses_path, std::ios::binary);
  if (f) {
    std::lock_guard<std::mutex> lock(mtx_kf_);
    uint32_t n = static_cast<uint32_t>(keyframes_.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto &kf : keyframes_) {
      double tx = kf.pose.t_(0), ty = kf.pose.t_(1), tz = kf.pose.t_(2);
      f.write(reinterpret_cast<const char*>(&tx), sizeof(double));
      f.write(reinterpret_cast<const char*>(&ty), sizeof(double));
      f.write(reinterpret_cast<const char*>(&tz), sizeof(double));
    }
    LOG(INFO) << GREEN << " ---> [STD] Saved " << n << " keyframe poses." << RESET;
  }
}

}  // namespace LI2Sup
