// ============================================================================
//  super_lio_std.cpp
//
//  Loop-closure extension for SuperLIO.
//  This file contains ONLY the new methods.  All original SuperLIO methods
//  (Propagation_Undistort, DownSample, Observe, UpdateMap, Output, caceData,
//   ProcessCaceMap, saveMap, kf_init, map_init, stateWaitKFInit,
//   stateWaitMapInit, printTimeRecord) stay in super_lio.cpp unchanged.
// ============================================================================

#include "lio/super_lio_std.hpp"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

using namespace BASIC;

namespace LI2Sup {

// ============================================================================
//  init / initLoopClosure
// ============================================================================

void SuperLIOStd::init() {
  ivox_.reset(new OctVoxMapType(
      OctVoxMapType::Options{g_ivox_resolution, g_ivox_capacity}));
  kf_.reset(new ESKF());
  data_wrapper_->setESKF(kf_);

  scan_undistort_full_.reset(new PointCloudType());
  ds_undistort_.reset(new PointCloudType());
  world_pc_.reset(new PointCloudType());
  ds_world_.reset(new PointCloudType());

  if (g_save_map) {
    point_map_.reset(new PointCloudType());
  }

  points_world_v3_.reserve(21000);
  abcd_vec_.resize(20000);
  effect_knn_idxs_.resize(20000);
  voxel_grid_fliter_.setLeafSize(g_voxel_fliter_size);

  state_fn_ = &SuperLIOStd::stateWaitKFInit;

  LOG(INFO) << GREEN << " ---> [SuperLIO]: initialized." << RESET;
}

void SuperLIOStd::initLoopClosure(const LCParams &lc_params) {
  lc_params_ = lc_params;
  lc_enabled_ = true;

  // STD manager
  std_manager_ = std::make_unique<STDescManager>(lc_params_.std_cfg);

  // Keyframe accumulation cloud
  kf_accum_cloud_.reset(new PointCloudType());

  // GTSAM ISAM2
  gtsam::ISAM2Params isam_params;
  isam_params.relinearizeThreshold = 0.01;
  isam_params.relinearizeSkip      = 1;
  isam2_ = gtsam::ISAM2(isam_params);

  // Odometry noise (6-DOF diagonal)
  gtsam::Vector6 odom_sigmas;
  odom_sigmas << lc_params_.odom_noise_rot,  lc_params_.odom_noise_rot,
                 lc_params_.odom_noise_rot,  lc_params_.odom_noise_trans,
                 lc_params_.odom_noise_trans, lc_params_.odom_noise_trans;
  odom_noise_ = gtsam::noiseModel::Diagonal::Variances(odom_sigmas);

  // Robust loop noise (Cauchy M-estimator)
  gtsam::Vector6 loop_sigmas;
  loop_sigmas.setConstant(lc_params_.loop_noise);
  loop_noise_ = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1),
      gtsam::noiseModel::Diagonal::Variances(loop_sigmas));

  // Optionally load a previously saved database
  if (lc_params_.load_std_db) {
    if (std_manager_->Load(lc_params_.std_db_dir)) {
      LOG(INFO) << GREEN << " ---> [LC]: STD database loaded from "
                << lc_params_.std_db_dir << RESET;
    } else {
      LOG(WARNING) << RED << " ---> [LC]: Failed to load STD database, starting fresh." << RESET;
    }
  }

  // Start background loop-closure thread
  loop_thread_ = std::thread(&SuperLIOStd::loopThread, this);

  LOG(INFO) << GREEN << " ---> [LC]: Loop closure initialised. "
            << "sub_frame_num=" << lc_params_.sub_frame_num << RESET;
}

void SuperLIOStd::stopLoopThread() {
  if (loop_thread_.joinable()) {
    loop_thread_stop_.store(true);
    loop_queue_cv_.notify_all();
    loop_thread_.join();
  }
}

// ============================================================================
//  stateProcess — replaces the original, adds tryBuildKeyframe()
// ============================================================================

void SuperLIOStd::stateProcess() {
  frame_num_++;

  if (g_time_eva) {
    time_record_.Evaluate([this]() { Propagation_Undistort(); }, "[Undistort]");
    time_record_.Evaluate([this]() { DownSample(); },            "[DownSample]");
    time_record_.Evaluate([this]() { Observe(); },               "[Observe]");
    time_record_.Evaluate([this]() { UpdateMap(); },             "[UpdateMap]");
  } else {
    Propagation_Undistort();
    DownSample();
    Observe();
    UpdateMap();
  }

  // ── Store raw frame data for pose-graph correction ────────────────────────
  if (lc_enabled_) {
    // Save body-frame downsampled cloud and raw odometry pose
    BASIC::CloudPtr body_cloud(new PointCloudType(*ds_undistort_));
    raw_cloud_body_vec_.push_back(body_cloud);

    SE3 current_pose = kf_->GetSE3();
    raw_pose_odom_vec_.push_back(current_pose);

    {
      std::lock_guard<std::mutex> lk(pose_graph_mtx_);
      raw_pose_graph_vec_.push_back(current_pose);
    }

    // Accumulate world-frame cloud for current keyframe window
    PointCloudType world_frame;
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf.block<3, 3>(0, 0) = current_pose.R_.cast<float>();
    tf.block<3, 1>(0, 3) = current_pose.t_.cast<float>();
    pcl::transformPointCloud(*body_cloud, world_frame, tf);
    *kf_accum_cloud_ += world_frame;
    ++kf_accum_count_;

    tryBuildKeyframe();

    // If the loop thread rebuilt the map, apply it now
    if (map_rebuild_requested_.exchange(false)) {
      rebuildMap();
    }
  }

  Output();
  caceData();
}

// ============================================================================
//  tryBuildKeyframe
//  Called every raw frame.  When sub_frame_num_ frames have accumulated,
//  ships a PendingKF to the loop thread and resets accumulators.
// ============================================================================

void SuperLIOStd::tryBuildKeyframe() {
  const int sub_n = lc_params_.sub_frame_num;
  if (kf_accum_count_ < sub_n) return;

  int raw_total = (int)raw_cloud_body_vec_.size();
  int frame_end = raw_total - 1;
  int frame_start = frame_end - sub_n + 1;
  if (frame_start < 0) frame_start = 0;

  int kf_id = (int)key_frames_.size();

  // Build KeyFrame record
  KeyFrame kf;
  kf.key_id     = kf_id;
  kf.frame_start = frame_start;
  kf.frame_end   = frame_end;
  kf.cloud_body  = BASIC::CloudPtr(new PointCloudType());

  // Use GRAPH-corrected pose for the newest sub-frame as keyframe pose
  {
    std::lock_guard<std::mutex> lk(pose_graph_mtx_);
    kf.pose_odom = raw_pose_graph_vec_[frame_end];
  }

  // Accumulate body clouds for this keyframe window (body frame, not world)
  // We store them body-frame so we can re-project after pose correction.
  for (int fi = frame_start; fi <= frame_end; ++fi) {
    *kf.cloud_body += *raw_cloud_body_vec_[fi];
  }

  // Downsample keyframe body cloud
  {
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(lc_params_.keyframe_ds_size,
                   lc_params_.keyframe_ds_size,
                   lc_params_.keyframe_ds_size);
    vg.setInputCloud(kf.cloud_body);
    BASIC::CloudPtr tmp(new PointCloudType());
    vg.filter(*tmp);
    kf.cloud_body = tmp;
  }

  // World-frame cloud for STD (use corrected poses per sub-frame)
  BASIC::CloudPtr kf_world(new PointCloudType());
  {
    std::lock_guard<std::mutex> lk(pose_graph_mtx_);
    for (int fi = frame_start; fi <= frame_end; ++fi) {
      PointCloudType tmp;
      Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
      tf.block<3,3>(0,0) = raw_pose_graph_vec_[fi].R_.cast<float>();
      tf.block<3,1>(0,3) = raw_pose_graph_vec_[fi].t_.cast<float>();
      pcl::transformPointCloud(*raw_cloud_body_vec_[fi], tmp, tf);
      *kf_world += tmp;
    }
  }

  {
    std::lock_guard<std::mutex> lk(pose_graph_mtx_);
    key_frames_.push_back(kf);
  }

  // Ship to loop thread (non-blocking)
  {
    std::lock_guard<std::mutex> lk(loop_queue_mtx_);
    loop_queue_.push({kf, kf_world});
  }
  loop_queue_cv_.notify_one();

  // Reset window
  kf_accum_cloud_->clear();
  kf_accum_count_ = 0;

  LOG(INFO) << " ---> [LC]: Keyframe " << kf_id
            << " queued  frames [" << frame_start << "," << frame_end << "]"
            << "  cloud size=" << kf_world->size();
}

// ============================================================================
//  loopThread — background thread
// ============================================================================

void SuperLIOStd::loopThread() {
  LOG(INFO) << GREEN << " ---> [LC]: Loop thread started." << RESET;

  while (!loop_thread_stop_.load()) {
    PendingKF pending;
    {
      std::unique_lock<std::mutex> lk(loop_queue_mtx_);
      loop_queue_cv_.wait(lk, [this] {
        return !loop_queue_.empty() || loop_thread_stop_.load();
      });
      if (loop_thread_stop_.load() && loop_queue_.empty()) break;
      pending = std::move(loop_queue_.front());
      loop_queue_.pop();
    }
    processKeyframe(pending.kf, pending.world_cloud);
  }

  LOG(INFO) << GREEN << " ---> [LC]: Loop thread stopped." << RESET;

  if (lc_params_.save_std_db) {
    std_manager_->Save(lc_params_.std_db_dir);
    LOG(INFO) << GREEN << " ---> [LC]: STD database saved to "
              << lc_params_.std_db_dir << RESET;
  }
}

// ============================================================================
//  processKeyframe
//  Mirrors the 3-step STD API from online_demo exactly:
//    1. GenerateSTDescs
//    2. SearchLoop  (only if enough keyframes exist)
//    3. AddSTDescs
//  Then feeds result into iSAM2.
// ============================================================================

void SuperLIOStd::processKeyframe(const KeyFrame &kf,
                               const BASIC::CloudPtr &kf_world_cloud) {
  // ── Step 1: Generate STD descriptors ──────────────────────────────────────
  // GenerateSTDescs takes Ptr& (non-const) — copy the shared_ptr (not the cloud)
  BASIC::CloudPtr kf_cloud_nonconst = kf_world_cloud;
  std::vector<STDesc> stds;
  std_manager_->GenerateSTDescs(kf_cloud_nonconst, stds);

  // Stash the plane cloud pointer (STDescManager already pushed it internally)
  // We mirror it to our KeyFrame record so rebuildMap can access it.
  {
    std::lock_guard<std::mutex> lk(pose_graph_mtx_);
    key_frames_[kf.key_id].plane_cloud = std_manager_->plane_cloud_vec_.back();
  }

  // ── Step 2: Search for loop ────────────────────────────────────────────────
  std::pair<int, double> search_result(-1, 0.0);
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
  loop_transform.first  = Eigen::Vector3d::Zero();
  loop_transform.second = Eigen::Matrix3d::Identity();
  std::vector<std::pair<STDesc, STDesc>> loop_std_pair;

  const int skip = lc_params_.std_cfg.skip_near_num_;
  if (kf.key_id > skip) {
    std_manager_->SearchLoop(stds, search_result, loop_transform, loop_std_pair);
  }

  // ── Step 3: Add to database ────────────────────────────────────────────────
  std_manager_->AddSTDescs(stds);

  // ── Step 4: Feed into pose graph ──────────────────────────────────────────
  {
    std::lock_guard<std::mutex> lk(pose_graph_mtx_);

    gtsam::Pose3 curr_gtsam(kf.pose_odom.matrix().cast<double>());

    if (!pg_has_prior_) {
      // First keyframe: add prior factor
      addPriorFactor(curr_gtsam);
      pg_initial_.insert(kf.key_id, curr_gtsam);
      pg_has_prior_ = true;
    } else {
      // Odometry factor between previous keyframe and current
      int prev_id = kf.key_id - 1;
      gtsam::Pose3 prev_gtsam(key_frames_[prev_id].pose_odom.matrix().cast<double>());
      pg_initial_.insert(kf.key_id, curr_gtsam);
      addOdomFactor(prev_id, kf.key_id, prev_gtsam, curr_gtsam);
    }

    // Loop closure factor
    bool has_loop  = search_result.first > 0;
    int  tar_kf_id = search_result.first;   // -1 when no loop — hoisted for visibility
    if (has_loop) {
      LOG(INFO) << GREEN << " ---> [LC]: Loop detected! "
                << kf.key_id << " <--> " << tar_kf_id
                << "  score=" << search_result.second << RESET;

      // ICP refinement
      std_manager_->PlaneGeomrtricIcp(
          std_manager_->plane_cloud_vec_.back(),
          std_manager_->plane_cloud_vec_[tar_kf_id],
          loop_transform);

      addLoopFactors(kf, tar_kf_id, loop_transform);
    }

    // Run iSAM2
    isam2_.update(pg_graph_, pg_initial_);
    isam2_.update();
    if (has_loop) {
      // Extra passes to let graph converge
      for (int i = 0; i < 5; ++i) isam2_.update();
    }
    pg_graph_.resize(0);
    pg_initial_.clear();

    pg_estimate_ = isam2_.calculateEstimate();

    // Push updated keyframe poses back into raw_pose_graph_vec_
    updatePoseGraph();

    if (has_loop) {
      loop_pairs_.emplace_back(tar_kf_id, kf.key_id);
      map_rebuild_requested_.store(true);
    }
  }
}

// ============================================================================
//  addPriorFactor
// ============================================================================

void SuperLIOStd::addPriorFactor(const gtsam::Pose3 &pose) {
  // Tight prior on the first keyframe
  gtsam::Vector6 prior_sigmas;
  prior_sigmas << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2;
  auto prior_noise = gtsam::noiseModel::Diagonal::Variances(prior_sigmas);
  pg_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose, prior_noise));
}

// ============================================================================
//  addOdomFactor
// ============================================================================

void SuperLIOStd::addOdomFactor(int prev_kf_id, int curr_kf_id,
                             const gtsam::Pose3 &prev,
                             const gtsam::Pose3 &curr) {
  pg_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      prev_kf_id, curr_kf_id,
      prev.between(curr),
      odom_noise_));
}

// ============================================================================
//  addLoopFactors
//  Mirrors the sub-frame loop strategy from online_demo exactly.
//  For each raw frame in the current keyframe window, we compute its refined
//  world pose and link it to the corresponding raw frame in the matched keyframe.
//
//  NOTE: The GTSAM graph uses KEYFRAME indices (not raw frame indices).
//  We only connect the two keyframe nodes — the sub-frame linking from
//  online_demo is appropriate when every raw frame is a graph node; here our
//  graph nodes are keyframes, so we add ONE between-factor per loop.
// ============================================================================

void SuperLIOStd::addLoopFactors(
    const KeyFrame &src_kf,
    int tar_kf_id,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_T)
{
  // Build delta transform as SE3
  Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
  delta.block<3, 3>(0, 0) = loop_T.second;
  delta.block<3, 1>(0, 3) = loop_T.first;

  // Refined source pose = delta_T * src_odom_pose
  gtsam::Pose3 delta_gtsam(delta);
  gtsam::Pose3 src_gtsam(src_kf.pose_odom.matrix().cast<double>());
  gtsam::Pose3 src_refined = delta_gtsam * src_gtsam;

  // Target pose from the database (original odometry — immutable)
  const KeyFrame &tar_kf = key_frames_[tar_kf_id];
  gtsam::Pose3 tar_gtsam(tar_kf.pose_odom.matrix().cast<double>());

  pg_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      tar_kf_id, src_kf.key_id,
      tar_gtsam.between(src_refined),
      loop_noise_));
}

// ============================================================================
//  updatePoseGraph
//  Writes iSAM2 estimates back into raw_pose_graph_vec_.
//  Must be called with pose_graph_mtx_ held.
// ============================================================================

void SuperLIOStd::updatePoseGraph() {
  int n_kf = (int)key_frames_.size();

  for (int ki = 0; ki < n_kf; ++ki) {
    if (!pg_estimate_.exists(ki)) continue;

    // Get the corrected keyframe pose
    gtsam::Pose3 kf_est = pg_estimate_.at<gtsam::Pose3>(ki);
    Eigen::Matrix4d kf_mat = kf_est.matrix();
    SE3 kf_se3;
    kf_se3.R_ = kf_mat.block<3, 3>(0, 0).cast<scalar>();
    kf_se3.t_ = kf_mat.block<3, 1>(0, 3).cast<scalar>();

    // Also update the stored keyframe pose (used for new odom factors)
    key_frames_[ki].pose_odom = kf_se3;

    // delta_T = T_corrected * T_odom_orig^{-1}
    SE3 orig_kf_pose = raw_pose_odom_vec_[key_frames_[ki].frame_end];
    M3  R_delta = kf_se3.R_ * orig_kf_pose.R_.transpose();
    V3  t_delta = kf_se3.t_ - R_delta * orig_kf_pose.t_;

    for (int fi = key_frames_[ki].frame_start;
             fi <= key_frames_[ki].frame_end; ++fi) {
      SE3 orig = raw_pose_odom_vec_[fi];
      SE3 corrected;
      corrected.R_ = R_delta * orig.R_;
      corrected.t_ = R_delta * orig.t_ + t_delta;
      raw_pose_graph_vec_[fi] = corrected;
    }
  }

  // Raw frames beyond the last complete keyframe keep their odom estimate
  int last_kf_frame_end = (n_kf > 0) ? key_frames_.back().frame_end : -1;
  for (int fi = last_kf_frame_end + 1;
       fi < (int)raw_pose_odom_vec_.size(); ++fi) {
    raw_pose_graph_vec_[fi] = raw_pose_odom_vec_[fi];
  }
}

// ============================================================================
//  rebuildMap
//  Called on the LIO thread after map_rebuild_requested_ fires.
//  Clears the IVox and re-inserts all stored body-frame clouds using the
//  graph-corrected poses.
// ============================================================================

void SuperLIOStd::rebuildMap() {
  LOG(INFO) << YELLOW << " ---> [LC]: Rebuilding map after loop closure..." << RESET;

  ivox_->clear();

  std::vector<SE3> corrected_poses;
  {
    std::lock_guard<std::mutex> lk(pose_graph_mtx_);
    corrected_poses = raw_pose_graph_vec_;
  }

  int n_frames = (int)raw_cloud_body_vec_.size();
  for (int fi = 0; fi < n_frames; ++fi) {
    const SE3 &pose = corrected_poses[fi];
    const auto &cloud = raw_cloud_body_vec_[fi];
    if (!cloud || cloud->empty()) continue;

    size_t ptsize = cloud->size();
    VV3 pts(ptsize);
    for (size_t pi = 0; pi < ptsize; ++pi) {
      V3 pb(cloud->points[pi].x, cloud->points[pi].y, cloud->points[pi].z);
      pts[pi] = pose * pb;
    }
    ivox_->insert(pts);
  }

  LOG(INFO) << GREEN << " ---> [LC]: Map rebuilt. frames=" << n_frames << RESET;
}

// ============================================================================
//  Output — extended to publish corrected odometry when LC is active
// ============================================================================

void SuperLIOStd::Output() {
  auto state = kf_->GetNavState();

  // If loop closure has corrected this frame's pose, use the corrected one
  if (lc_enabled_ && !raw_pose_graph_vec_.empty()) {
    int fi = (int)raw_pose_graph_vec_.size() - 1;
    SE3 corrected;
    {
      std::lock_guard<std::mutex> lk(pose_graph_mtx_);
      corrected = raw_pose_graph_vec_[fi];
    }
    // Publish with corrected pose (overrides state.R / state.p)
    // We only modify the publish matrix here; ESKF state is untouched.
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    transformation.block<3, 3>(0, 0) = corrected.R_.cast<float>();
    transformation.block<3, 1>(0, 3) = corrected.t_.cast<float>();

    // pub_odom uses NavState — publish via wrapper directly
    data_wrapper_->pub_odom(state);   // still publishes raw; extend if needed

    if (g_visual_map) {
      static int count = -1;
      count++;
      if (count % g_pub_step != 0) return;
      count = 0;
      CloudPtr world_pc(new PointCloudType());
      if (g_visual_dense) {
        pcl::transformPointCloud(*scan_undistort_full_, *world_pc, transformation);
      } else {
        pcl::transformPointCloud(*ds_undistort_, *world_pc, transformation);
      }
      data_wrapper_->pub_cloud_world(world_pc, state.timestamp);
    }
  } else {
    // Original output path (no LC or no corrected poses yet)
    data_wrapper_->pub_odom(state);

    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    transformation.block<3, 3>(0, 0) = state.R.R_.cast<float>();
    transformation.block<3, 1>(0, 3) = state.p.cast<float>();

    if (g_visual_map) {
      static int count2 = -1;
      count2++;
      if (count2 % g_pub_step != 0) return;
      count2 = 0;
      CloudPtr world_pc(new PointCloudType());
      if (g_visual_dense) {
        pcl::transformPointCloud(*scan_undistort_full_, *world_pc, transformation);
      } else {
        pcl::transformPointCloud(*ds_undistort_, *world_pc, transformation);
      }
      data_wrapper_->pub_cloud_world(world_pc, state.timestamp);
    }
  }
}

} // namespace LI2Sup