#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "ros/ROSWrapper.h"
#include "lio/super_lio_std.hpp"

using namespace LI2Sup;

int main(int argc, char** argv){
  rclcpp::init(argc, argv);

  ROSWrapper::Ptr data_wrapper = std::make_shared<ROSWrapper>();

  auto lio = std::make_shared<SuperLIOStd>();
  lio->setROSWrapper(data_wrapper);
  lio->init();

  // ── Loop closure params ───────────────────────────────────────────────────
  LCParams lc;

  /* pre-process */
  lc.std_cfg.ds_size_                    = 0.1;
  lc.std_cfg.maximum_corner_num_         = 100;

  /* key points / plane detection */
  lc.std_cfg.plane_detection_thre_       = 0.01;
  lc.std_cfg.plane_merge_normal_thre_    = 0.1;
  lc.std_cfg.plane_merge_dis_thre_       = 0.3;
  lc.std_cfg.voxel_size_                 = 2.0;
  lc.std_cfg.voxel_init_num_             = 10;
  lc.std_cfg.proj_image_resolution_      = 0.5;
  lc.std_cfg.proj_dis_min_               = 0.0;
  lc.std_cfg.proj_dis_max_               = 5.0;
  lc.std_cfg.corner_thre_                = 10.0;

  /* STD descriptor */
  lc.std_cfg.descriptor_near_num_        = 20;
  lc.std_cfg.descriptor_min_len_         = 2.0;
  lc.std_cfg.descriptor_max_len_         = 30.0;
  lc.std_cfg.non_max_suppression_radius_ = 2.0;
  lc.std_cfg.std_side_resolution_        = 0.2;

  /* candidate search / place recognition */
  lc.std_cfg.skip_near_num_              = 50;
  lc.std_cfg.candidate_num_             = 50;
  lc.std_cfg.sub_frame_num_              = 10;
  lc.std_cfg.vertex_diff_threshold_      = 0.5;
  lc.std_cfg.rough_dis_threshold_        = 0.01;
  lc.std_cfg.normal_threshold_           = 0.2;
  lc.std_cfg.dis_threshold_              = 0.5;
  lc.std_cfg.icp_threshold_              = 0.4;

  /* keyframe window — must match sub_frame_num_ */
  lc.sub_frame_num                       = 10;
  lc.keyframe_ds_size                    = 0.1f;

  /* database persistence */
  lc.save_std_db                         = true;
  lc.std_db_dir                          = "/tmp/std_db";

  lio->initLoopClosure(lc);
  // ─────────────────────────────────────────────────────────────────────────

  auto timer = data_wrapper->create_wall_timer(
    std::chrono::milliseconds(2),
    [lio]() { lio->process(); },
    data_wrapper->getSensorCallbackGroup()
  );

  rclcpp::spin(data_wrapper);

  lio->saveMap();
  lio->printTimeRecord();

  rclcpp::shutdown();
  return 0;
}