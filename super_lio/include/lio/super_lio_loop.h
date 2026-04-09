
#ifndef SUPER_LIO_LOOP_H_
#define SUPER_LIO_LOOP_H_

#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "super_lio.h"

namespace LI2Sup {

/**
 * SuperLIOLoop extends SuperLIO with LIO-SAM-style loop closure:
 *
 *  1. Keyframe management  — saves (pose, cloud) when robot moves far enough.
 *  2. Background thread    — runs ICP-based loop detection at loopClosureFrequency Hz.
 *  3. Detection            — radius search in keyframe-position history (distance + time gate).
 *  4. ICP registration     — current keyframe cloud vs. historical submap.
 *  5. ESKF correction      — applies the delta transform to the filter state.
 *  6. Map rebuild          — reconstructs the OctVoxMap from corrected keyframe poses.
 */
class SuperLIOLoop : public SuperLIO {
public:
  SuperLIOLoop()  = default;
  ~SuperLIOLoop();

  void init() override;

protected:
  void stateProcess() override;

private:
  // -----------------------------------------------------------------------
  // Keyframe
  // -----------------------------------------------------------------------
  struct KeyFrame {
    int          id;
    double       timestamp;
    BASIC::SE3   pose;    // IMU pose in world frame at capture time
    BASIC::CloudPtr cloud; // downsampled scan in body (IMU) frame
  };

  /** True when the robot has moved / rotated enough since the last keyframe. */
  bool shouldAddKeyframe(const BASIC::SE3& cur_pose) const;

  /** Copy ds_undistort_ and pose into the keyframe vector. */
  void saveKeyframe(double timestamp, const BASIC::SE3& pose);

  // -----------------------------------------------------------------------
  // Loop closure
  // -----------------------------------------------------------------------
  struct LoopResult {
    int        cur_id;           ///< index of the current (drifted) keyframe
    int        ref_id;           ///< index of the historical reference keyframe
    BASIC::SE3 corrected_pose;   ///< corrected world pose of keyframes_[cur_id]
  };

  /** Background thread: calls performLoopClosure() at the configured frequency. */
  void loopClosureThread();

  /**
   * Radius-search for a historical keyframe close in space but far in time.
   * Populates *cur_id / *pre_id and returns true on success.
   */
  bool detectLoopClosureDistance(int* cur_id, int* pre_id);

  /**
   * Build a downsampled world-frame submap from keyframes [center-half, center+half].
   * Caller must already hold mtx_keyframes_.
   */
  BASIC::CloudPtr buildSubmap(int center_id, int half_window);

  /** Run detection + ICP; push LoopResult to loop_queue_ on success. */
  void performLoopClosure();

  /**
   * Called at the top of stateProcess() (main thread).
   * Pops one LoopResult from loop_queue_, applies the delta to the ESKF state,
   * corrects all keyframe poses at/after cur_id, and rebuilds the voxel map.
   */
  void applyLoopCorrection();

  /** Reset ivox_ and re-insert all keyframe clouds with their (corrected) poses. */
  void rebuildMap();

  /** Publish loop-constraint markers to RViz. */
  void visualizeLoopClosure();

  // -----------------------------------------------------------------------
  // Data
  // -----------------------------------------------------------------------
  std::vector<KeyFrame> keyframes_;
  std::mutex            mtx_keyframes_;

  std::queue<LoopResult> loop_queue_;
  std::mutex             mtx_loop_;

  std::map<int, int>    loop_index_container_;   ///< cur_id -> ref_id, dedup guard
  std::mutex            mtx_loop_index_;

  std::thread      loop_thread_;
  std::atomic<bool> loop_thread_running_{false};

  bool       has_last_keyframe_pose_{false};
  BASIC::SE3 last_keyframe_pose_;

  // -----------------------------------------------------------------------
  // ROS publishers (created on data_wrapper_ node in init())
  // -----------------------------------------------------------------------
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_loop_markers_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr        pub_history_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr        pub_icp_cloud_;
};

} // namespace LI2Sup

#endif  // SUPER_LIO_LOOP_H_
