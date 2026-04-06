

#include "lio/params.h"

using namespace std;
using namespace BASIC;

namespace LI2Sup{

  const std::string g_root_dir = std::string(ROOT);
  std::atomic<bool> g_flag_run = true; 
  bool g_flg_map_init = true;

  /// evaluation
  bool g_time_eva = false;

  bool   g_save_map;
  bool   g_if_filter; 
  string g_save_map_dir;
  string g_map_name;
  float  g_map_ds_size;
  int    g_pcd_save_interval;
  
  string g_imu_topic;
  string g_lidar_topic;

  double g_lidar_rotation_x = 0.0;
  double g_lidar_rotation_y = 0.0;
  double g_lidar_rotation_z = 0.0;

  int    g_lidar_type;
  float  g_blind2;
  float  g_maxrange2;
  int    g_filter_rate;
  bool   g_enable_downsample;
  float  g_voxel_fliter_size;

  int    g_imu_type;
  double g_gravity_norm = 9.7946;
  double g_imu_na;
  double g_imu_ng;
  double g_imu_nba;
  double g_imu_nbg;

  SE3 g_lidar_imu;
  SE3 g_odom_robo;
  M3  g_lidar_robo_yaw;

  /// hash_map
  std::size_t g_ivox_capacity = 100000;
  float       g_ivox_resolution = 0.5;

  /// kf
  int g_kf_type = 1;                // 1: ESKF, 2: InESKF
  int g_kf_max_iterations = 4;
  bool g_kf_align_gravity = true;
  double g_kf_quit_eps;

  /// submap 
  double g_submap_resolution;
  int    g_submap_capacity;

  /// output
  bool g_2_robot    = false;
  bool g_2_plan_env_world = false; 
  bool g_2_plan_env_body  = false;
  bool g_2_ml_map = false;
  bool g_visual_map = true;
  bool g_visual_dense = false;
  int  g_pub_step;

  /// for planner
  bool g_planner_enable;

  ResidualType g_residual_type = PROB;

  /// for relocation
  bool g_update_map = false;
  double g_init_px, g_init_py, g_init_pz, g_init_roll, g_init_pitch, g_init_yaw;

  /// local sliding-window map
  bool   g_local_map_enable        = false;
  double g_local_map_radius        = 50.0;
  int    g_local_map_prune_interval = 10;

  /// for loop closure
  bool   g_loop_closure_enable    = false;
  double g_loop_closure_frequency = 1.0;
  double g_loop_keyframe_add_dist  = 1.0;
  double g_loop_keyframe_add_angle = 0.2;
  double g_loop_search_radius     = 15.0;
  double g_loop_search_time_diff  = 30.0;
  int    g_loop_search_num        = 25;
  double g_loop_fitness_score     = 0.3;
  double g_loop_icp_leaf_size     = 0.4;

  /// STD descriptor parameters
  double g_std_ds_size                    = 0.5;
  int    g_std_maximum_corner_num         = 100;
  double g_std_plane_merge_normal_thre    = 0.1;
  double g_std_plane_merge_dis_thre       = 0.1;
  double g_std_plane_detection_thre       = 0.01;
  double g_std_voxel_size                 = 1.0;
  int    g_std_voxel_init_num             = 10;
  double g_std_proj_image_resolution      = 0.5;
  double g_std_proj_dis_min               = 0.2;
  double g_std_proj_dis_max               = 5.0;
  double g_std_corner_thre                = 10.0;
  int    g_std_descriptor_near_num        = 10;
  double g_std_descriptor_min_len         = 1.0;
  double g_std_descriptor_max_len         = 10.0;
  double g_std_non_max_suppression_radius = 3.0;
  double g_std_std_side_resolution        = 0.2;
  int    g_std_candidate_num              = 50;
  double g_std_rough_dis_threshold        = 0.03;
  double g_std_vertex_diff_threshold      = 0.7;
  double g_std_icp_threshold              = 0.5;
  double g_std_normal_threshold           = 0.1;
  double g_std_dis_threshold              = 0.3;
  int    g_std_sub_frame_num              = 10;

}