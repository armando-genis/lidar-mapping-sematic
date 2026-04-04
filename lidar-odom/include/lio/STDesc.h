#pragma once

// ============================================================================
//  STDesc.h  —  ROS2-compatible adaptation of the STD paper implementation
//  Original: Yuan et al., "STD: Stable Triangle Descriptor for 3D place
//            recognition", ICRA 2023. https://arxiv.org/abs/2209.12435
//
//  Changes from original:
//    - Removed ros/ros.h, ros::NodeHandle  → plain C++ config struct
//    - Removed visualization_msgs (ROS1)   → optional ROS2 publisher separated
//    - Added Save() / Load() for database persistence across sessions
//    - Kept ALL core math identical to original
// ============================================================================

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>
#include <mutex>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <filesystem>

#define HASH_P 116101
#define MAX_N 10000000000
#define MAX_FRAME_N 20000

// ============================================================================
//  ConfigSetting — replaces ros::NodeHandle param reading
//  All values match original paper defaults.
//  Wire these to your yaml in params.h / ROSWrapper just like other g_ params.
// ============================================================================
typedef struct ConfigSetting {
  /* point cloud pre-process */
  int    stop_skip_enable_     = 0;
  double ds_size_              = 0.5;
  int    maximum_corner_num_   = 30;

  /* key points */
  double plane_merge_normal_thre_  = 0.1;
  double plane_merge_dis_thre_     = 0.01;
  double plane_detection_thre_     = 0.01;
  double voxel_size_               = 1.0;
  int    voxel_init_num_           = 10;
  double proj_image_resolution_    = 0.5;
  double proj_dis_min_             = 0.2;
  double proj_dis_max_             = 5.0;
  double corner_thre_              = 10.0;

  /* STD descriptor */
  int    descriptor_near_num_          = 10;
  double descriptor_min_len_           = 1.0;
  double descriptor_max_len_           = 10.0;
  double non_max_suppression_radius_   = 3.0;
  double std_side_resolution_          = 0.2;

  /* place recognition */
  int    skip_near_num_          = 50;
  int    candidate_num_          = 50;
  int    sub_frame_num_          = 10;
  double rough_dis_threshold_    = 0.03;
  double vertex_diff_threshold_  = 0.7;
  double icp_threshold_          = 0.5;
  double normal_threshold_       = 0.1;
  double dis_threshold_          = 0.3;
} ConfigSetting;

// ============================================================================
//  Core data structures — identical to original
// ============================================================================

typedef struct STDesc {
  Eigen::Vector3d side_length_;    // sorted short→long, quantized by resolution
  Eigen::Vector3d angle_;          // pairwise vertex-normal dot products * 5
  Eigen::Vector3d center_;         // world-frame centroid of triangle
  unsigned int    frame_id_;

  Eigen::Vector3d vertex_A_;
  Eigen::Vector3d vertex_B_;
  Eigen::Vector3d vertex_C_;
  Eigen::Vector3d vertex_attached_; // intensity at each vertex
} STDesc;

typedef struct Plane {
  pcl::PointXYZINormal p_center_;
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Matrix3d covariance_;
  float radius_          = 0;
  float min_eigen_value_ = 1;
  float intercept_       = 0;
  int   id_              = 0;
  int   sub_plane_num_   = 0;
  int   points_size_     = 0;
  bool  is_plane_        = false;
} Plane;

typedef struct STDMatchList {
  std::vector<std::pair<STDesc, STDesc>> match_list_;
  std::pair<int, int>                    match_id_;
  double                                 mean_dis_;
} STDMatchList;

// ============================================================================
//  Voxel location hash key
// ============================================================================
class VOXEL_LOC {
public:
  int64_t x, y, z;
  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}
  bool operator==(const VOXEL_LOC &o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct M_POINT {
  float xyz[3];
  float intensity;
  int   count = 0;
};

template <> struct std::hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + s.x;
  }
};

// STDesc hash key (quantized side lengths)
class STDesc_LOC {
public:
  int64_t x, y, z, a, b, c;
  STDesc_LOC(int64_t vx=0,int64_t vy=0,int64_t vz=0,
             int64_t va=0,int64_t vb=0,int64_t vc=0)
      : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}
  bool operator==(const STDesc_LOC &o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

template <> struct std::hash<STDesc_LOC> {
  int64_t operator()(const STDesc_LOC &s) const {
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + s.x;
  }
};

// ============================================================================
//  OctoTree — plane detection via recursive octree subdivision
// ============================================================================
class OctoTree {
public:
  ConfigSetting config_setting_;
  std::vector<Eigen::Vector3d> voxel_points_;
  Plane   *plane_ptr_;
  int      layer_;
  int      octo_state_;   // 0 = leaf, 1 = has children
  int      merge_num_ = 0;
  bool     is_project_ = false;
  std::vector<Eigen::Vector3d> proj_normal_vec_;

  bool      is_check_connect_[6];
  bool      connect_[6];
  OctoTree *connect_tree_[6];

  bool      is_publish_ = false;
  OctoTree *leaves_[8];
  double    voxel_center_[3];
  float     quater_length_;
  bool      init_octo_;

  OctoTree(const ConfigSetting &cfg) : config_setting_(cfg) {
    voxel_points_.clear();
    octo_state_ = 0;
    layer_      = 0;
    init_octo_  = false;
    for (int i = 0; i < 8; i++) leaves_[i] = nullptr;
    for (int i = 0; i < 6; i++) {
      is_check_connect_[i] = false;
      connect_[i]          = false;
      connect_tree_[i]     = nullptr;
    }
    plane_ptr_ = new Plane;
  }

  void init_plane();
  void init_octo_tree();
};

// ============================================================================
//  Ceres cost function for plane-to-plane ICP refinement
// ============================================================================
struct PlaneSolver {
  PlaneSolver(Eigen::Vector3d cp, Eigen::Vector3d cn,
              Eigen::Vector3d tp, Eigen::Vector3d tn)
      : curr_point(cp), curr_normal(cn), target_point(tp), target_normal(tn) {}

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Quaternion<T> q_w{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T,3,1> t_w{t[0], t[1], t[2]};
    Eigen::Matrix<T,3,1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
    Eigen::Matrix<T,3,1> pw = q_w * cp + t_w;
    Eigen::Matrix<T,3,1> pt(T(target_point.x()), T(target_point.y()), T(target_point.z()));
    Eigen::Matrix<T,3,1> n(T(target_normal.x()), T(target_normal.y()), T(target_normal.z()));
    residual[0] = n.dot(pw - pt);
    return true;
  }

  static ceres::CostFunction *Create(Eigen::Vector3d cp, Eigen::Vector3d cn,
                                      Eigen::Vector3d tp, Eigen::Vector3d tn) {
    return new ceres::AutoDiffCostFunction<PlaneSolver,1,4,3>(
        new PlaneSolver(cp, cn, tp, tn));
  }

  Eigen::Vector3d curr_point, curr_normal, target_point, target_normal;
};

// ============================================================================
//  Free functions
// ============================================================================
void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size);

bool attach_greater_sort(std::pair<double,int> a, std::pair<double,int> b);

pcl::PointXYZI    vec2point(const Eigen::Vector3d &vec);
Eigen::Vector3d   point2vec(const pcl::PointXYZI &pi);

// ============================================================================
//  STDescManager
// ============================================================================
class STDescManager {
public:
  STDescManager() = default;
  explicit STDescManager(const ConfigSetting &cfg)
      : config_setting_(cfg), current_frame_id_(0) {}

  ConfigSetting config_setting_;
  unsigned int  current_frame_id_;

  // Hash-table database: key = quantized side-length bucket
  std::unordered_map<STDesc_LOC, std::vector<STDesc>> data_base_;

  // Per-frame plane clouds (required for geometric verification)
  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> plane_cloud_vec_;

  // Optional stored raw/corner clouds
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>        key_cloud_vec_;
  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr>   corner_cloud_vec_;

  // ── Main API ─────────────────────────────────────────────────────────────

  // Generate descriptors from a world-frame point cloud
  void GenerateSTDescs(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                       std::vector<STDesc> &stds_vec);

  // Search for loop candidate. Returns frame_id (-1 = no match) and score.
  void SearchLoop(const std::vector<STDesc> &stds_vec,
                  std::pair<int, double> &loop_result,
                  std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
                  std::vector<std::pair<STDesc,STDesc>> &loop_std_pair);

  // Add descriptors to database (call after GenerateSTDescs per keyframe)
  void AddSTDescs(const std::vector<STDesc> &stds_vec);

  // Plane-to-plane Ceres ICP refinement
  void PlaneGeomrtricIcp(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform);

  // ── Persistence (new — not in original paper code) ───────────────────────

  // Save database + plane clouds to directory.
  // Creates: <dir>/std_descs.bin  and  <dir>/std_planes_{i}.bin
  bool Save(const std::string &dir) const;

  // Load database + plane clouds from directory.
  bool Load(const std::string &dir);

private:
  void init_voxel_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::unordered_map<VOXEL_LOC, OctoTree*> &voxel_map);

  void build_connection(std::unordered_map<VOXEL_LOC, OctoTree*> &feat_map);

  void getPlane(const std::unordered_map<VOXEL_LOC, OctoTree*> &voxel_map,
                pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud);

  void corner_extractor(
      std::unordered_map<VOXEL_LOC, OctoTree*> &voxel_map,
      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points);

  void extract_corner(
      const Eigen::Vector3d &proj_center,
      const Eigen::Vector3d  proj_normal,
      const std::vector<Eigen::Vector3d> proj_points,
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points);

  void non_maxi_suppression(
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points);

  void build_stdesc(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points,
      std::vector<STDesc> &stds_vec);

  void candidate_selector(
      const std::vector<STDesc> &stds_vec,
      std::vector<STDMatchList> &candidate_matcher_vec);

  void candidate_verify(
      const STDMatchList &candidate_matcher,
      double &verify_score,
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
      std::vector<std::pair<STDesc,STDesc>> &sucess_match_vec);

  void triangle_solver(std::pair<STDesc,STDesc> &std_pair,
                       Eigen::Vector3d &t, Eigen::Matrix3d &rot);

  double plane_geometric_verify(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
      const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform);

  // ── Serialization helpers ─────────────────────────────────────────────────
  static void write_vec3d(std::ofstream &f, const Eigen::Vector3d &v);
  static void read_vec3d (std::ifstream &f,       Eigen::Vector3d &v);
  static void write_stdesc(std::ofstream &f, const STDesc &d);
  static void read_stdesc (std::ifstream &f,       STDesc &d);
};
