
#ifndef SUPER_LIO_LOOP_H_
#define SUPER_LIO_LOOP_H_

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_map>

// GTSAM – pose graph backend
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

// PCL utilities for ICP and KD-tree
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

// ROS2 visualization
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "super_lio.h"
#include "lio/STDesc.h"


namespace LI2Sup {

// ---------------------------------------------------------------------------
// KeyFrame: the atomic unit stored for loop closure.
//   - cloud  : body-frame downsampled scan (allows re-projection with any pose)
//   - pose   : world-frame pose at capture time (updated after graph optimisation)
// ---------------------------------------------------------------------------
struct KeyFrame {
  BASIC::SE3     pose;
  double         timestamp{0.0};
  BASIC::CloudPtr cloud;          // body-frame downsampled PointXYZI cloud
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ---------------------------------------------------------------------------
// SuperLIOLoop
//
// Extends SuperLIO with:
//   1. Keyframe management (saved at configurable distance/angle thresholds)
//   2. GTSAM iSAM2 pose graph (odometry factors + loop-closure factors)
//   3. Background loop-detection thread (KD-tree radius search + ICP verify)
//   4. Full iVox map rebuild after every accepted loop closure
// ---------------------------------------------------------------------------
class SuperLIOLoop : public SuperLIO {
public:
  SuperLIOLoop();
  ~SuperLIOLoop();

  void init() override;

protected:
  // Called every frame from stateProcess() ─────────────────────────────────
  void DownSample() override;
  void UpdateMap() override;
  void Output()   override;

private:
  // ── Keyframe management ─────────────────────────────────────────────────
  bool shouldAddKeyframe() const;
  void addKeyframe();

  // ── GTSAM factor graph ──────────────────────────────────────────────────
  // addOdomFactor / addLoopFactor enqueue work; optimizeGraph commits it.
  void addOdomFactor();
  void addLoopFactor();
  void optimizeGraph(bool loop_closed);

  // ── Loop closure (runs in background thread) ─────────────────────────────
  void loopClosureThread();
  bool performLoopClosure();
  bool detectLoop(int* curID, int* preID);
  void buildLocalMap(BASIC::CloudPtr& out, int centerIdx, int halfWidth);

  // ── Map rebuild ──────────────────────────────────────────────────────────
  // Called on the main thread after a loop factor is accepted and optimised.
  void rebuildMap();

  // ── Sliding-window local map (odometry mode) ─────────────────────────────
  // Clears the iVox and re-inserts only keyframes within g_local_map_radius.
  void pruneLocalMap();

  // ── Helpers ──────────────────────────────────────────────────────────────
  gtsam::Pose3 toGtsamPose(const BASIC::SE3& se3)      const;
  BASIC::SE3   fromGtsamPose(const gtsam::Pose3& pose) const;
  void         publishLoopConstraints();
  void         publishCorrectedPath();

  // ── STDescManager ────────────────────────────────────────────────────────
  void saveSTDDatabase();

  STDescManager                    std_manager_;
  std::vector<std::vector<STDesc>> kf_stds_;     // per-keyframe descriptors
  ConfigSetting                    std_cfg_;      // STD parameters


private:
  // ── Keyframe store (guarded by mtx_kf_) ─────────────────────────────────
  mutable std::mutex  mtx_kf_;
  std::vector<KeyFrame> keyframes_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr kf_positions_;  // world-XYZ for KD-tree

  // ── GTSAM ────────────────────────────────────────────────────────────────
  gtsam::NonlinearFactorGraph gtsam_graph_;
  gtsam::Values               gtsam_init_vals_;
  gtsam::ISAM2*               isam_{nullptr};
  gtsam::Values               isam_estimate_;
  int                         kf_count_{0};       // == GTSAM node index

  // Previous keyframe data needed for odometry factor
  gtsam::Pose3  last_kf_gtsam_;
  BASIC::SE3    last_kf_se3_;
  bool          has_last_kf_{false};
  bool          new_keyframe_{false};   // set by UpdateMap, consumed by Output
  int           prune_counter_{0};      // counts new keyframes since last prune

  // ── Loop queues (written by loop thread, consumed by main thread) ─────────
  std::mutex                                               mtx_loop_;
  std::vector<std::pair<int, int>>                         loop_index_queue_;
  std::vector<gtsam::Pose3>                                loop_pose_queue_;
  std::vector<gtsam::noiseModel::Diagonal::shared_ptr>     loop_noise_queue_;
  std::unordered_map<int, int>                             loop_history_;   // curID -> preID

  // ── Rebuild signal ───────────────────────────────────────────────────────
  std::atomic<bool> flag_rebuild_{false};

  // ── Loop thread ──────────────────────────────────────────────────────────
  std::thread       loop_thread_;
  std::atomic<bool> loop_thread_stop_{false};

  // ── ICP voxel filter (only used by the loop thread) ─────────────────────
  pcl::VoxelGrid<BASIC::PointType> icp_ds_filter_;

  // ── Lidar rotation ────────────────────────────────────────────────────────
  float              sensor_rotation_x_{0.f};
  float              sensor_rotation_y_{0.f};
  float              sensor_rotation_z_{0.f};
  Eigen::Matrix4f    rotation_matrix_{Eigen::Matrix4f::Identity()};

  // ── ROS2 publishers (created at init time) ───────────────────────────────
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_loop_constraints_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                  pub_corrected_path_;
};

}  // namespace LI2Sup

#endif  // SUPER_LIO_LOOP_H_
