#ifndef SUPER_LIO_STD_H_
#define SUPER_LIO_STD_H_

#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

// GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "lio/super_lio.h"
#include "lio/STDesc.h"

namespace LI2Sup {

// ============================================================================
//  Loop closure parameters
// ============================================================================
struct LCParams {
  // Keyframe policy
  int   sub_frame_num        = 10;
  float keyframe_ds_size     = 0.5f;

  // STD descriptor (forwarded to ConfigSetting)
  ConfigSetting std_cfg;

  // GTSAM noise
  double odom_noise_rot      = 1e-6;
  double odom_noise_trans    = 1e-4;
  double loop_noise          = 1e-1;

  // Save/load database
  bool        save_std_db    = false;
  bool        load_std_db    = false;
  std::string std_db_dir     = "/tmp/std_db";
};

// ============================================================================
//  Per-keyframe data stored for map correction
// ============================================================================
struct KeyFrame {
  int                              key_id;
  int                              frame_start;
  int                              frame_end;
  BASIC::SE3                       pose_odom;
  BASIC::CloudPtr                  cloud_body;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud;
};

// ============================================================================
//  SuperLIOStd: SuperLIO with loop closure (inherits base LIO pipeline)
// ============================================================================
class SuperLIOStd : public SuperLIO {
public:
  SuperLIOStd() {}
  virtual ~SuperLIOStd() { stopLoopThread(); }

  // Override init to also set up LC accumulators
  virtual void init() override;

  // Call once after init() to enable loop closure
  void initLoopClosure(const LCParams &lc_params);

protected:
  // ── Overridden pipeline steps ─────────────────────────────────────────────
  virtual void stateProcess() override;
  virtual void Output() override;

  // ── Loop closure internals ────────────────────────────────────────────────
  void tryBuildKeyframe();
  void processKeyframe(const KeyFrame &kf, const BASIC::CloudPtr &kf_world_cloud);

  void addPriorFactor(const gtsam::Pose3 &pose);
  void addOdomFactor(int prev_kf_id, int curr_kf_id,
                     const gtsam::Pose3 &prev, const gtsam::Pose3 &curr);
  void addLoopFactors(const KeyFrame &src_kf, int tar_kf_id,
                      const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_T);
  void updatePoseGraph();
  void rebuildMap();

  void loopThread();
  void stopLoopThread();

  // ── Loop closure members ───────────────────────────────────────────────────
  LCParams lc_params_;
  bool     lc_enabled_ = false;

  std::vector<BASIC::CloudPtr>    raw_cloud_body_vec_;
  std::vector<BASIC::SE3>         raw_pose_odom_vec_;
  std::vector<BASIC::SE3>         raw_pose_graph_vec_;

  std::vector<KeyFrame>           key_frames_;

  BASIC::CloudPtr                 kf_accum_cloud_;
  int                             kf_accum_count_ = 0;

  std::unique_ptr<STDescManager>  std_manager_;

  gtsam::NonlinearFactorGraph     pg_graph_;
  gtsam::Values                   pg_initial_;
  gtsam::ISAM2                    isam2_;
  gtsam::Values                   pg_estimate_;
  bool                            pg_has_prior_ = false;

  gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
  gtsam::noiseModel::Base::shared_ptr     loop_noise_;

  std::vector<std::pair<int, int>> loop_pairs_;

  struct PendingKF {
    KeyFrame         kf;
    BASIC::CloudPtr  world_cloud;
  };
  std::thread                      loop_thread_;
  std::mutex                       loop_queue_mtx_;
  std::condition_variable          loop_queue_cv_;
  std::queue<PendingKF>            loop_queue_;
  std::atomic<bool>                loop_thread_stop_{false};

  std::mutex                       pose_graph_mtx_;
  std::atomic<bool>                map_rebuild_requested_{false};
};

} // namespace LI2Sup

#endif  // SUPER_LIO_STD_H_
