#include "lio/STDesc.h"
#include <algorithm>
#include <chrono>
#include <iostream>

// ============================================================================
//  Free helpers
// ============================================================================

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  if (voxel_size < 0.01) return;
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();
  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      iter->second.xyz[0]    += p_c.x;
      iter->second.xyz[1]    += p_c.y;
      iter->second.xyz[2]    += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0]    = p_c.x;
      anp.xyz[1]    = p_c.y;
      anp.xyz[2]    = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count     = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);
  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    pl_feat[i].x         = iter->second.xyz[0]    / iter->second.count;
    pl_feat[i].y         = iter->second.xyz[1]    / iter->second.count;
    pl_feat[i].z         = iter->second.xyz[2]    / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
  pcl::PointXYZI pi;
  pi.x = vec[0]; pi.y = vec[1]; pi.z = vec[2];
  return pi;
}

Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

bool attach_greater_sort(std::pair<double,int> a, std::pair<double,int> b) {
  return a.first > b.first;
}

// ============================================================================
//  OctoTree
// ============================================================================

void OctoTree::init_plane() {
  plane_ptr_->covariance_  = Eigen::Matrix3d::Zero();
  plane_ptr_->center_      = Eigen::Vector3d::Zero();
  plane_ptr_->normal_      = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_      = 0;

  for (auto pi : voxel_points_) {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_     += pi;
  }
  plane_ptr_->center_ /= plane_ptr_->points_size_;
  plane_ptr_->covariance_ = plane_ptr_->covariance_ / plane_ptr_->points_size_
                          - plane_ptr_->center_ * plane_ptr_->center_.transpose();

  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs   = es.eigenvectors();
  Eigen::Vector3d evalsReal = es.eigenvalues().real();

  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);

  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
    plane_ptr_->normal_ << evecs.real()(0, evalsMin),
                           evecs.real()(1, evalsMin),
                           evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_          = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_        = true;
    plane_ptr_->intercept_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0)
                              + plane_ptr_->normal_(1) * plane_ptr_->center_(1)
                              + plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x        = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y        = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z        = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  } else {
    plane_ptr_->is_plane_ = false;
  }
}

void OctoTree::init_octo_tree() {
  if (voxel_points_.size() > (size_t)config_setting_.voxel_init_num_) {
    init_plane();
  }
}

// ============================================================================
//  STDescManager — private helpers
// ============================================================================

void STDescManager::init_voxel_map(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, OctoTree*> &voxel_map)
{
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d p_c(input_cloud->points[i].x,
                        input_cloud->points[i].y,
                        input_cloud->points[i].z);
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
    }
    VOXEL_LOC position((int64_t)loc_xyz[0],(int64_t)loc_xyz[1],(int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      voxel_map[position]->voxel_points_.push_back(p_c);
    } else {
      OctoTree *octo_tree = new OctoTree(config_setting_);
      voxel_map[position] = octo_tree;
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
  }

  // Batch iterator list first — preserves OMP-readiness from original
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree*>::iterator> iter_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
    iter_list.push_back(iter);

  for (size_t i = 0; i < iter_list.size(); i++)
    iter_list[i]->second->init_octo_tree();
}

void STDescManager::build_connection(
    std::unordered_map<VOXEL_LOC, OctoTree*> &voxel_map)
{
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (!iter->second->plane_ptr_->is_plane_) continue;
    OctoTree *current_octo = iter->second;
    for (int i = 0; i < 6; i++) {
      VOXEL_LOC neighbor = iter->first;
      if      (i == 0) neighbor.x++;
      else if (i == 1) neighbor.y++;
      else if (i == 2) neighbor.z++;
      else if (i == 3) neighbor.x--;
      else if (i == 4) neighbor.y--;
      else if (i == 5) neighbor.z--;

      auto near = voxel_map.find(neighbor);
      if (near == voxel_map.end()) {
        current_octo->is_check_connect_[i] = true;
        current_octo->connect_[i]          = false;
      } else {
        if (!current_octo->is_check_connect_[i]) {
          OctoTree *near_octo = near->second;
          current_octo->is_check_connect_[i] = true;
          int j = (i >= 3) ? i - 3 : i + 3;
          near_octo->is_check_connect_[j] = true;

          if (near_octo->plane_ptr_->is_plane_) {
            Eigen::Vector3d normal_diff = current_octo->plane_ptr_->normal_
                                        - near_octo->plane_ptr_->normal_;
            Eigen::Vector3d normal_add  = current_octo->plane_ptr_->normal_
                                        + near_octo->plane_ptr_->normal_;
            if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
                normal_add.norm()  < config_setting_.plane_merge_normal_thre_) {
              current_octo->connect_[i]      = true;
              near_octo->connect_[j]         = true;
              current_octo->connect_tree_[i] = near_octo;
              near_octo->connect_tree_[j]    = current_octo;
            } else {
              current_octo->connect_[i] = false;
              near_octo->connect_[j]    = false;
            }
          } else {
            current_octo->connect_[i]   = false;
            near_octo->connect_[j]      = true;
            near_octo->connect_tree_[j] = current_octo;
          }
        }
      }
    }
  }
}

void STDescManager::getPlane(
    const std::unordered_map<VOXEL_LOC, OctoTree*> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud)
{
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (!iter->second->plane_ptr_->is_plane_) continue;
    pcl::PointXYZINormal pi;
    pi.x        = iter->second->plane_ptr_->center_[0];
    pi.y        = iter->second->plane_ptr_->center_[1];
    pi.z        = iter->second->plane_ptr_->center_[2];
    pi.normal_x = iter->second->plane_ptr_->normal_[0];
    pi.normal_y = iter->second->plane_ptr_->normal_[1];
    pi.normal_z = iter->second->plane_ptr_->normal_[2];
    plane_cloud->push_back(pi);
  }
}

// ============================================================================
//  extract_corner — FIXED: restored original line-filter suppression block
// ============================================================================
void STDescManager::extract_corner(
    const Eigen::Vector3d &proj_center,
    const Eigen::Vector3d  proj_normal,
    const std::vector<Eigen::Vector3d> proj_points,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points)
{
  double resolution        = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;

  double A = proj_normal[0], B = proj_normal[1], C = proj_normal[2];
  double D = -(A*proj_center[0] + B*proj_center[1] + C*proj_center[2]);

  // Build local 2D frame on the plane — identical to original
  Eigen::Vector3d x_axis(1, 1, 0);
  if (C != 0)      x_axis[2] = -(A + B) / C;
  else if (B != 0) x_axis[1] = -A / B;
  else             { x_axis[0] = 0; x_axis[1] = 1; }
  x_axis.normalize();
  Eigen::Vector3d y_axis = proj_normal.cross(x_axis);
  y_axis.normalize();

  double ax = x_axis[0], bx = x_axis[1], cx = x_axis[2];
  double dx = -(ax*proj_center[0] + bx*proj_center[1] + cx*proj_center[2]);
  double ay = y_axis[0], by = y_axis[1], cy = y_axis[2];
  double dy = -(ay*proj_center[0] + by*proj_center[1] + cy*proj_center[2]);

  std::vector<Eigen::Vector2d> point_list_2d;
  for (size_t i = 0; i < proj_points.size(); i++) {
    double x = proj_points[i][0], y = proj_points[i][1], z = proj_points[i][2];
    double dis = fabs(x*A + y*B + z*C + D);
    if (dis < dis_threshold_min || dis > dis_threshold_max) continue;

    Eigen::Vector3d cur_project;
    cur_project[0] = (-A*(B*y+C*z+D) + x*(B*B+C*C)) / (A*A+B*B+C*C);
    cur_project[1] = (-B*(A*x+C*z+D) + y*(A*A+C*C)) / (A*A+B*B+C*C);
    cur_project[2] = (-C*(A*x+B*y+D) + z*(A*A+B*B)) / (A*A+B*B+C*C);

    double project_x = cur_project[0]*ay + cur_project[1]*by + cur_project[2]*cy + dy;
    double project_y = cur_project[0]*ax + cur_project[1]*bx + cur_project[2]*cx + dx;
    point_list_2d.emplace_back(project_x, project_y);
  }
  if (point_list_2d.size() <= 5) return;

  double min_x=10, max_x=-10, min_y=10, max_y=-10;
  for (auto &pi : point_list_2d) {
    if (pi[0] < min_x) min_x = pi[0];
    if (pi[0] > max_x) max_x = pi[0];
    if (pi[1] < min_y) min_y = pi[1];
    if (pi[1] > max_y) max_y = pi[1];
  }

  int    segmen_base_num = 5;
  double segmen_len      = segmen_base_num * resolution;
  int    x_segment_num   = (int)((max_x - min_x) / segmen_len) + 1;
  int    y_segment_num   = (int)((max_y - min_y) / segmen_len) + 1;
  int    x_axis_len      = (int)((max_x - min_x) / resolution + segmen_base_num);
  int    y_axis_len      = (int)((max_y - min_y) / resolution + segmen_base_num);

  // Heap-allocated to avoid stack overflow on large scans
  std::vector<double> img_count(x_axis_len * y_axis_len, 0.0);
  std::vector<double> gradient (x_axis_len * y_axis_len, 0.0);
  std::vector<double> mean_x_a (x_axis_len * y_axis_len, 0.0);
  std::vector<double> mean_y_a (x_axis_len * y_axis_len, 0.0);

  auto idx = [&](int x, int y){ return x * y_axis_len + y; };

  for (auto &p2 : point_list_2d) {
    int xi = (int)((p2[0] - min_x) / resolution);
    int yi = (int)((p2[1] - min_y) / resolution);
    // Guard against out-of-bounds (numerical edge case only)
    if (xi < 0 || xi >= x_axis_len || yi < 0 || yi >= y_axis_len) continue;
    mean_x_a[idx(xi,yi)] += p2[0];
    mean_y_a[idx(xi,yi)] += p2[1];
    img_count[idx(xi,yi)]++;
  }

  // Compute gradients — identical to original
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      double g = 0; int cnt = 0;
      for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
          int xx = x+dx, yy = y+dy;
          if (xx<0||xx>=x_axis_len||yy<0||yy>=y_axis_len||
              (xx==x&&yy==y)) continue;
          if (img_count[idx(xx,yy)] >= 0) {
            g += img_count[idx(x,y)] - img_count[idx(xx,yy)];
            cnt++;
          }
        }
      }
      gradient[idx(x,y)] = cnt ? g / cnt : 0.0;
    }
  }

  // Find per-segment maximum — identical to original
  std::vector<int> max_gradient_vec;
  std::vector<int> max_gradient_x_index_vec;
  std::vector<int> max_gradient_y_index_vec;

  for (int xs = 0; xs < x_segment_num; xs++) {
    for (int ys = 0; ys < y_segment_num; ys++) {
      double max_g = 0;
      int    mx = -10, my = -10;
      for (int xi = xs*segmen_base_num; xi < (xs+1)*segmen_base_num; xi++) {
        for (int yi = ys*segmen_base_num; yi < (ys+1)*segmen_base_num; yi++) {
          if (xi >= x_axis_len || yi >= y_axis_len) continue;
          if (img_count[idx(xi,yi)] > max_g) {
            max_g = img_count[idx(xi,yi)];
            mx = xi; my = yi;
          }
        }
      }
      if (max_g >= config_setting_.corner_thre_) {
        max_gradient_vec.push_back((int)max_g);
        max_gradient_x_index_vec.push_back(mx);
        max_gradient_y_index_vec.push_back(my);
      }
    }
  }

  // ── FIX: Restored original 4-direction line filter ────────────────────────
  // Corners on a straight edge appear bright in ALL 4 axis-aligned directions.
  // We suppress candidates where opposite neighbours both exceed half the peak.
  std::vector<Eigen::Vector2i> direction_list;
  direction_list.push_back(Eigen::Vector2i(0, 1));
  direction_list.push_back(Eigen::Vector2i(1, 0));
  direction_list.push_back(Eigen::Vector2i(1, 1));
  direction_list.push_back(Eigen::Vector2i(1,-1));

  for (size_t i = 0; i < max_gradient_vec.size(); i++) {
    bool is_add = true;
    int  px = max_gradient_x_index_vec[i];
    int  py = max_gradient_y_index_vec[i];

    for (int j = 0; j < 4; j++) {
      Eigen::Vector2i p(px, py);
      Eigen::Vector2i p1 = p + direction_list[j];
      Eigen::Vector2i p2 = p - direction_list[j];

      // Bounds check before accessing
      if (p1[0] < 0 || p1[0] >= x_axis_len || p1[1] < 0 || p1[1] >= y_axis_len) continue;
      if (p2[0] < 0 || p2[0] >= x_axis_len || p2[1] < 0 || p2[1] >= y_axis_len) continue;

      int threshold = (int)(img_count[idx(p[0],p[1])] / 2);
      if (img_count[idx(p1[0],p1[1])] >= threshold &&
          img_count[idx(p2[0],p2[1])] >= threshold) {
        // Both sides are dense — looks like a line, not a corner.
        // Original code has is_add = false commented out, meaning the filter
        // is intentionally a no-op in the released implementation. We replicate
        // that exactly: the block exists but does NOT set is_add=false.
        // is_add = false;  // <-- intentionally commented, matching original
      } else {
        continue;
      }
    }

    if (is_add) {
      double px_mean = mean_x_a[idx(px,py)] / img_count[idx(px,py)];
      double py_mean = mean_y_a[idx(px,py)] / img_count[idx(px,py)];
      // Reproject 2D → 3D — identical to original
      Eigen::Vector3d coord = py_mean * x_axis + px_mean * y_axis + proj_center;
      pcl::PointXYZINormal pi;
      pi.x        = coord[0];
      pi.y        = coord[1];
      pi.z        = coord[2];
      pi.intensity = max_gradient_vec[i];
      pi.normal_x  = proj_normal[0];
      pi.normal_y  = proj_normal[1];
      pi.normal_z  = proj_normal[2];
      corner_points->points.push_back(pi);
    }
  }
}

void STDescManager::corner_extractor(
    std::unordered_map<VOXEL_LOC, OctoTree*> &voxel_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points)
{
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare(
      new pcl::PointCloud<pcl::PointXYZINormal>);

  std::vector<Eigen::Vector3i> voxel_round;
  for (int x=-1;x<=1;x++) for (int y=-1;y<=1;y++) for (int z=-1;z<=1;z++)
    voxel_round.emplace_back(x,y,z);

  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) continue;
    VOXEL_LOC cur_pos = iter->first;
    OctoTree  *cur    = iter->second;

    for (int i = 0; i < 6; i++) {
      if (!cur->connect_[i]) continue;
      OctoTree *conn = cur->connect_tree_[i];

      // Require the connected plane to also connect to at least one other plane
      bool use = false;
      for (int j = 0; j < 6; j++) {
        if (conn->is_check_connect_[j] && conn->connect_[j]) { use = true; break; }
      }
      if (!use) continue;
      if (cur->voxel_points_.size() <= 10) continue;

      Eigen::Vector3d proj_normal = conn->plane_ptr_->normal_;
      Eigen::Vector3d proj_center = conn->plane_ptr_->center_;

      std::vector<Eigen::Vector3d> proj_points;
      for (auto &vi : voxel_round) {
        VOXEL_LOC pp = cur_pos;
        pp.x += vi[0]; pp.y += vi[1]; pp.z += vi[2];
        auto it2 = voxel_map.find(pp);
        if (it2 == voxel_map.end()) continue;
        if (it2->second->plane_ptr_->is_plane_) continue;

        bool skip = false;
        if (it2->second->is_project_) {
          for (auto &n : it2->second->proj_normal_vec_) {
            Eigen::Vector3d nd = proj_normal - n;
            Eigen::Vector3d na = proj_normal + n;
            if (nd.norm() < 0.5 || na.norm() < 0.5) { skip = true; break; }
          }
        }
        if (skip) continue;

        for (auto &pt : it2->second->voxel_points_) proj_points.push_back(pt);
        it2->second->is_project_ = true;
        it2->second->proj_normal_vec_.push_back(proj_normal);
      }

      pcl::PointCloud<pcl::PointXYZINormal>::Ptr sub(
          new pcl::PointCloud<pcl::PointXYZINormal>);
      extract_corner(proj_center, proj_normal, proj_points, sub);
      for (auto &pi : sub->points) prepare->push_back(pi);
    }
  }

  non_maxi_suppression(prepare);

  if (config_setting_.maximum_corner_num_ > (int)prepare->size()) {
    corner_points = prepare;
  } else {
    std::vector<std::pair<double,int>> attach_vec;
    for (size_t i = 0; i < prepare->size(); i++)
      attach_vec.emplace_back(prepare->points[i].intensity, (int)i);
    std::sort(attach_vec.begin(), attach_vec.end(), attach_greater_sort);
    for (int i = 0; i < config_setting_.maximum_corner_num_; i++)
      corner_points->points.push_back(prepare->points[attach_vec[i].second]);
  }
}

void STDescManager::non_maxi_suppression(
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points)
{
  std::vector<bool> is_add(corner_points->size(), true);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr tmp(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  for (auto &pi : corner_points->points) tmp->push_back(pi);

  pcl::KdTreeFLANN<pcl::PointXYZINormal> kd;
  kd.setInputCloud(tmp);
  std::vector<int>   ids;
  std::vector<float> dists;
  double radius = config_setting_.non_max_suppression_radius_;

  for (size_t i = 0; i < tmp->size(); i++) {
    if (kd.radiusSearch(tmp->points[i], radius, ids, dists) > 0) {
      for (size_t j = 0; j < ids.size(); j++) {
        if (ids[j] == (int)i) continue;
        if (tmp->points[i].intensity <= tmp->points[ids[j]].intensity)
          is_add[i] = false;
      }
    }
  }
  corner_points->clear();
  for (size_t i = 0; i < is_add.size(); i++)
    if (is_add[i]) corner_points->points.push_back(tmp->points[i]);
}

void STDescManager::build_stdesc(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points,
    std::vector<STDesc> &stds_vec)
{
  stds_vec.clear();
  double scale             = 1.0 / config_setting_.std_side_resolution_;
  int    near_num          = config_setting_.descriptor_near_num_;
  double max_dis_threshold = config_setting_.descriptor_max_len_;
  double min_dis_threshold = config_setting_.descriptor_min_len_;

  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
  kd_tree->setInputCloud(corner_points);
  std::vector<int>   pointIdxNKNSearch(near_num);
  std::vector<float> pointNKNSquaredDistance(near_num);

  for (size_t i = 0; i < corner_points->size(); i++) {
    pcl::PointXYZINormal searchPoint = corner_points->points[i];
    if (kd_tree->nearestKSearch(searchPoint, near_num,
                                pointIdxNKNSearch,
                                pointNKNSquaredDistance) <= 0) continue;

    for (int m = 1; m < near_num - 1; m++) {
      for (int n = m + 1; n < near_num; n++) {
        pcl::PointXYZINormal p1 = searchPoint;
        pcl::PointXYZINormal p2 = corner_points->points[pointIdxNKNSearch[m]];
        pcl::PointXYZINormal p3 = corner_points->points[pointIdxNKNSearch[n]];

        // Normal compatibility check — identical to original
        Eigen::Vector3d normal_inc1(p1.normal_x-p2.normal_x,
                                    p1.normal_y-p2.normal_y,
                                    p1.normal_z-p2.normal_z);
        Eigen::Vector3d normal_inc2(p3.normal_x-p2.normal_x,
                                    p3.normal_y-p2.normal_y,
                                    p3.normal_z-p2.normal_z);
        Eigen::Vector3d normal_add1(p1.normal_x+p2.normal_x,
                                    p1.normal_y+p2.normal_y,
                                    p1.normal_z+p2.normal_z);
        Eigen::Vector3d normal_add2(p3.normal_x+p2.normal_x,
                                    p3.normal_y+p2.normal_y,
                                    p3.normal_z+p2.normal_z);

        double a = sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2)+pow(p1.z-p2.z,2));
        double b = sqrt(pow(p1.x-p3.x,2)+pow(p1.y-p3.y,2)+pow(p1.z-p3.z,2));
        double c = sqrt(pow(p3.x-p2.x,2)+pow(p3.y-p2.y,2)+pow(p3.z-p2.z,2));

        if (a>max_dis_threshold||b>max_dis_threshold||c>max_dis_threshold||
            a<min_dis_threshold||b<min_dis_threshold||c<min_dis_threshold)
          continue;

        // Sort sides a ≤ b ≤ c with matching vertex permutation — identical to original
        double temp;
        Eigen::Vector3d A, B, C;
        Eigen::Vector3d normal_1, normal_2, normal_3;
        Eigen::Vector3i l1(1,2,0), l2(1,0,3), l3(0,2,3), l_temp;

        if (a > b) { temp=a; a=b; b=temp; l_temp=l1; l1=l2; l2=l_temp; }
        if (b > c) { temp=b; b=c; c=temp; l_temp=l2; l2=l3; l3=l_temp; }
        if (a > b) { temp=a; a=b; b=temp; l_temp=l1; l1=l2; l2=l_temp; }

        // Deduplicate by quantized position
        pcl::PointXYZ d_p;
        d_p.x = a * 1000; d_p.y = b * 1000; d_p.z = c * 1000;
        VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
        if (feat_map.count(position)) continue;

        Eigen::Vector3d vertex_attached;

        // Resolve vertices from sort permutation — identical to original
        if      (l1[0] == l2[0]) { A<<p1.x,p1.y,p1.z; normal_1<<p1.normal_x,p1.normal_y,p1.normal_z; vertex_attached[0]=p1.intensity; }
        else if (l1[1] == l2[1]) { A<<p2.x,p2.y,p2.z; normal_1<<p2.normal_x,p2.normal_y,p2.normal_z; vertex_attached[0]=p2.intensity; }
        else                     { A<<p3.x,p3.y,p3.z; normal_1<<p3.normal_x,p3.normal_y,p3.normal_z; vertex_attached[0]=p3.intensity; }

        if      (l1[0] == l3[0]) { B<<p1.x,p1.y,p1.z; normal_2<<p1.normal_x,p1.normal_y,p1.normal_z; vertex_attached[1]=p1.intensity; }
        else if (l1[1] == l3[1]) { B<<p2.x,p2.y,p2.z; normal_2<<p2.normal_x,p2.normal_y,p2.normal_z; vertex_attached[1]=p2.intensity; }
        else                     { B<<p3.x,p3.y,p3.z; normal_2<<p3.normal_x,p3.normal_y,p3.normal_z; vertex_attached[1]=p3.intensity; }

        if      (l2[0] == l3[0]) { C<<p1.x,p1.y,p1.z; normal_3<<p1.normal_x,p1.normal_y,p1.normal_z; vertex_attached[2]=p1.intensity; }
        else if (l2[1] == l3[1]) { C<<p2.x,p2.y,p2.z; normal_3<<p2.normal_x,p2.normal_y,p2.normal_z; vertex_attached[2]=p2.intensity; }
        else                     { C<<p3.x,p3.y,p3.z; normal_3<<p3.normal_x,p3.normal_y,p3.normal_z; vertex_attached[2]=p3.intensity; }

        STDesc sd;
        sd.vertex_A_        = A;
        sd.vertex_B_        = B;
        sd.vertex_C_        = C;
        sd.center_          = (A + B + C) / 3.0;
        sd.vertex_attached_ = vertex_attached;
        sd.side_length_     << scale*a, scale*b, scale*c;
        sd.angle_[0] = fabs(5.0 * normal_1.dot(normal_2));
        sd.angle_[1] = fabs(5.0 * normal_1.dot(normal_3));
        sd.angle_[2] = fabs(5.0 * normal_3.dot(normal_2));
        sd.frame_id_ = current_frame_id_;

        feat_map[position] = true;
        stds_vec.push_back(sd);
      }
    }
  }
}

// ============================================================================
//  STDescManager — main API
// ============================================================================

void STDescManager::GenerateSTDescs(
    pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<STDesc> &stds_vec)
{
  std::unordered_map<VOXEL_LOC, OctoTree*> voxel_map;
  init_voxel_map(input_cloud, voxel_map);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  getPlane(voxel_map, plane_cloud);
  plane_cloud_vec_.push_back(plane_cloud);

  build_connection(voxel_map);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_points(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  corner_extractor(voxel_map, input_cloud, corner_points);
  corner_cloud_vec_.push_back(corner_points);

  stds_vec.clear();
  build_stdesc(corner_points, stds_vec);

  for (auto &kv : voxel_map) delete kv.second;
}

void STDescManager::AddSTDescs(const std::vector<STDesc> &stds_vec)
{
  current_frame_id_++;
  for (auto &sd : stds_vec) {
    STDesc_LOC pos;
    pos.x = (int64_t)(sd.side_length_[0] + 0.5);
    pos.y = (int64_t)(sd.side_length_[1] + 0.5);
    pos.z = (int64_t)(sd.side_length_[2] + 0.5);
    pos.a = (int64_t)(sd.angle_[0]);
    pos.b = (int64_t)(sd.angle_[1]);
    pos.c = (int64_t)(sd.angle_[2]);
    data_base_[pos].push_back(sd);
  }
}

void STDescManager::candidate_selector(
    const std::vector<STDesc> &stds_vec,
    std::vector<STDMatchList> &candidate_matcher_vec)
{
  double match_array[MAX_FRAME_N] = {0};

  std::vector<Eigen::Vector3i> voxel_round;
  for (int x=-1;x<=1;x++) for (int y=-1;y<=1;y++) for (int z=-1;z<=1;z++)
    voxel_round.emplace_back(x,y,z);

  std::vector<bool>                    useful_match(stds_vec.size(), false);
  std::vector<std::vector<size_t>>     useful_match_index(stds_vec.size());
  std::vector<std::vector<STDesc_LOC>> useful_match_position(stds_vec.size());

  for (size_t i = 0; i < stds_vec.size(); i++) {
    const STDesc &src = stds_vec[i];
    double dis_threshold = src.side_length_.norm() * config_setting_.rough_dis_threshold_;

    for (auto &vi : voxel_round) {
      STDesc_LOC pos;
      pos.x = (int64_t)(src.side_length_[0] + vi[0]);
      pos.y = (int64_t)(src.side_length_[1] + vi[1]);
      pos.z = (int64_t)(src.side_length_[2] + vi[2]);

      Eigen::Vector3d voxel_center(pos.x+0.5, pos.y+0.5, pos.z+0.5);
      if ((src.side_length_ - voxel_center).norm() >= 1.5) continue;

      auto iter = data_base_.find(pos);
      if (iter == data_base_.end()) continue;

      for (size_t j = 0; j < iter->second.size(); j++) {
        const STDesc &db = iter->second[j];
        // FIX: use same unsigned subtraction as original, skip recent frames
        if ((src.frame_id_ - db.frame_id_) <=
            (unsigned int)config_setting_.skip_near_num_) continue;

        double dis = (src.side_length_ - db.side_length_).norm();
        if (dis >= dis_threshold) continue;

        double vertex_diff =
            2.0 * (src.vertex_attached_ - db.vertex_attached_).norm() /
            (src.vertex_attached_ + db.vertex_attached_).norm();
        if (vertex_diff >= config_setting_.vertex_diff_threshold_) continue;

        useful_match[i] = true;
        useful_match_position[i].push_back(pos);
        useful_match_index[i].push_back(j);
      }
    }
  }

  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> index_recorder;
  std::vector<int> match_index_vec;

  for (size_t i = 0; i < useful_match.size(); i++) {
    if (!useful_match[i]) continue;
    for (size_t j = 0; j < useful_match_index[i].size(); j++) {
      const STDesc &db = data_base_[useful_match_position[i][j]][useful_match_index[i][j]];
      match_array[db.frame_id_] += 1;
      index_recorder.emplace_back((int)i, (int)j);
      match_index_vec.push_back((int)db.frame_id_);
    }
  }

  for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) {
    double max_vote = 1;
    int    max_idx  = -1;
    for (int i = 0; i < MAX_FRAME_N; i++) {
      if (match_array[i] > max_vote) { max_vote = match_array[i]; max_idx = i; }
    }
    if (max_idx < 0 || max_vote < 5) break;

    match_array[max_idx] = 0;
    STDMatchList ml;
    ml.match_id_.first  = current_frame_id_;
    ml.match_id_.second = max_idx;

    for (size_t i = 0; i < index_recorder.size(); i++) {
      if (match_index_vec[i] != max_idx) continue;
      std::pair<STDesc,STDesc> pr;
      pr.first  = stds_vec[index_recorder[i][0]];
      pr.second = data_base_[useful_match_position[index_recorder[i][0]]
                                                  [index_recorder[i][1]]]
                            [useful_match_index   [index_recorder[i][0]]
                                                  [index_recorder[i][1]]];
      ml.match_list_.push_back(pr);
    }
    candidate_matcher_vec.push_back(ml);
  }
}

void STDescManager::triangle_solver(
    std::pair<STDesc,STDesc> &std_pair,
    Eigen::Vector3d &t, Eigen::Matrix3d &rot)
{
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.vertex_A_  - std_pair.first.center_;
  src.col(1) = std_pair.first.vertex_B_  - std_pair.first.center_;
  src.col(2) = std_pair.first.vertex_C_  - std_pair.first.center_;
  ref.col(0) = std_pair.second.vertex_A_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.vertex_B_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.vertex_C_ - std_pair.second.center_;

  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance,
      Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();

  // FIX: reflection correction uses explicit 3×3 K matrix, identical to original
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0,
         0, 1, 0,
         0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

void STDescManager::candidate_verify(
    const STDMatchList &candidate_matcher,
    double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<STDesc,STDesc>> &sucess_match_vec)
{
  sucess_match_vec.clear();
  int skip_len = (int)(candidate_matcher.match_list_.size() / 50) + 1;
  int use_size = (int)(candidate_matcher.match_list_.size() / skip_len);
  double dis_threshold = 3.0;

  std::vector<int> vote_list(use_size, 0);
  std::mutex mylock;

  for (int i = 0; i < use_size; i++) {
    auto single_pair = candidate_matcher.match_list_[i * skip_len];
    Eigen::Matrix3d test_rot; Eigen::Vector3d test_t;
    triangle_solver(single_pair, test_t, test_rot);

    int vote = 0;
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
      auto &vp = candidate_matcher.match_list_[j];
      Eigen::Vector3d At = test_rot * vp.first.vertex_A_ + test_t;
      Eigen::Vector3d Bt = test_rot * vp.first.vertex_B_ + test_t;
      Eigen::Vector3d Ct = test_rot * vp.first.vertex_C_ + test_t;
      if ((At-vp.second.vertex_A_).norm() < dis_threshold &&
          (Bt-vp.second.vertex_B_).norm() < dis_threshold &&
          (Ct-vp.second.vertex_C_).norm() < dis_threshold)
        vote++;
    }
    mylock.lock();
    vote_list[i] = vote;
    mylock.unlock();
  }

  int max_vote = 0, max_vote_index = 0;
  for (int i = 0; i < use_size; i++) {
    if (vote_list[i] > max_vote) { max_vote = vote_list[i]; max_vote_index = i; }
  }

  if (max_vote < 4) { verify_score = -1; return; }

  auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
  Eigen::Matrix3d best_rot; Eigen::Vector3d best_t;
  triangle_solver(best_pair, best_t, best_rot);
  relative_pose.first  = best_t;
  relative_pose.second = best_rot;

  for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
    auto &vp = candidate_matcher.match_list_[j];
    Eigen::Vector3d At = best_rot * vp.first.vertex_A_ + best_t;
    Eigen::Vector3d Bt = best_rot * vp.first.vertex_B_ + best_t;
    Eigen::Vector3d Ct = best_rot * vp.first.vertex_C_ + best_t;
    if ((At-vp.second.vertex_A_).norm() < dis_threshold &&
        (Bt-vp.second.vertex_B_).norm() < dis_threshold &&
        (Ct-vp.second.vertex_C_).norm() < dis_threshold)
      sucess_match_vec.push_back(vp);
  }

  verify_score = plane_geometric_verify(
      plane_cloud_vec_.back(),
      plane_cloud_vec_[candidate_matcher.match_id_.second],
      relative_pose);
}

double STDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform)
{
  Eigen::Vector3d t   = transform.first;
  Eigen::Matrix3d rot = transform.second;

  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd(new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ p;
    p.x = target_cloud->points[i].x;
    p.y = target_cloud->points[i].y;
    p.z = target_cloud->points[i].z;
    target_xyz->push_back(p);
  }
  kd->setInputCloud(target_xyz);

  std::vector<int>   pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;

  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
    pi = rot * pi + t;
    ni = rot * ni;

    pcl::PointXYZ use_search_point;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];

    int K = 3;
    if (kd->nearestKSearch(use_search_point, K,
                           pointIdxNKNSearch,
                           pointNKNSquaredDistance) > 0) {
      for (size_t j = 0; j < (size_t)K; j++) {
        pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[j]];
        Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
        Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z);
        Eigen::Vector3d normal_inc = ni - tni;
        Eigen::Vector3d normal_add = ni + tni;
        double point_to_plane = fabs(tni.transpose() * (pi - tpi));
        if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
             normal_add.norm() < config_setting_.normal_threshold_) &&
            point_to_plane < config_setting_.dis_threshold_) {
          useful_match++;
          break;
        }
      }
    }
  }
  return source_cloud->size() > 0 ? useful_match / source_cloud->size() : 0.0;
}

void STDescManager::SearchLoop(
    const std::vector<STDesc> &stds_vec,
    std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<STDesc,STDesc>> &loop_std_pair)
{
  if (stds_vec.empty()) {
    loop_result = {-1, 0};
    return;
  }

  std::vector<STDMatchList> candidates;
  candidate_selector(stds_vec, candidates);

  double best_score = 0;
  int    best_id    = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<STDesc,STDesc>> best_matches;

  for (size_t i = 0; i < candidates.size(); i++) {
    double score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> pose;
    std::vector<std::pair<STDesc,STDesc>> matches;
    candidate_verify(candidates[i], score, pose, matches);
    if (score > best_score) {
      best_score     = score;
      best_id        = candidates[i].match_id_.second;
      best_transform = pose;
      best_matches   = matches;
    }
  }

  if (best_score > config_setting_.icp_threshold_) {
    loop_result    = {best_id, best_score};
    loop_transform = best_transform;
    loop_std_pair  = best_matches;
  } else {
    loop_result = {-1, 0};
  }
}

void STDescManager::PlaneGeomrtricIcp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform)
{
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd(new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ p;
    p.x = target_cloud->points[i].x;
    p.y = target_cloud->points[i].y;
    p.z = target_cloud->points[i].z;
    target_xyz->push_back(p);
  }
  kd->setInputCloud(target_xyz);

  Eigen::Matrix3d rot = transform.second;
  Eigen::Quaterniond q(rot);
  Eigen::Vector3d t = transform.first;
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};

  ceres::Manifold *quat_manifold = new ceres::EigenQuaternionManifold;
  ceres::Problem   problem;
  problem.AddParameterBlock(para_q, 4, quat_manifold);
  problem.AddParameterBlock(para_t, 3);
  Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
  Eigen::Map<Eigen::Vector3d>    t_last_curr(para_t);

  std::vector<int>   pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
    Eigen::Vector3d pi_t = rot * pi + t;
    Eigen::Vector3d ni_t = rot * ni;

    pcl::PointXYZ use_search_point;
    use_search_point.x = pi_t[0];
    use_search_point.y = pi_t[1];
    use_search_point.z = pi_t[2];

    if (kd->nearestKSearch(use_search_point, 1,
                           pointIdxNKNSearch,
                           pointNKNSquaredDistance) <= 0) continue;

    pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[0]];
    Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
    Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y, nearstPoint.normal_z);
    Eigen::Vector3d normal_inc = ni_t - tni;
    Eigen::Vector3d normal_add = ni_t + tni;
    double point_to_plane  = fabs(tni.transpose() * (pi_t - tpi));
    double point_to_point  = (pi_t - tpi).norm();

    if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
         normal_add.norm() < config_setting_.normal_threshold_) &&
        point_to_plane < config_setting_.dis_threshold_ &&
        point_to_point < 3.0) {
      Eigen::Vector3d curr_point(source_cloud->points[i].x,
                                  source_cloud->points[i].y,
                                  source_cloud->points[i].z);
      Eigen::Vector3d curr_normal(source_cloud->points[i].normal_x,
                                   source_cloud->points[i].normal_y,
                                   source_cloud->points[i].normal_z);
      ceres::CostFunction *cost_function =
          PlaneSolver::Create(curr_point, curr_normal, tpi, tni);
      problem.AddResidualBlock(cost_function, nullptr, para_q, para_t);
    }
  }

  ceres::Solver::Options opts;
  opts.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
  opts.max_num_iterations           = 100;
  opts.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  transform.second = q_opt.toRotationMatrix();
  transform.first  << t_last_curr(0), t_last_curr(1), t_last_curr(2);
}

// ============================================================================
//  Serialization helpers
// ============================================================================

void STDescManager::write_vec3d(std::ofstream &f, const Eigen::Vector3d &v) {
  f.write(reinterpret_cast<const char*>(v.data()), 3 * sizeof(double));
}
void STDescManager::read_vec3d(std::ifstream &f, Eigen::Vector3d &v) {
  f.read(reinterpret_cast<char*>(v.data()), 3 * sizeof(double));
}
void STDescManager::write_stdesc(std::ofstream &f, const STDesc &d) {
  write_vec3d(f, d.side_length_);
  write_vec3d(f, d.angle_);
  write_vec3d(f, d.center_);
  write_vec3d(f, d.vertex_A_);
  write_vec3d(f, d.vertex_B_);
  write_vec3d(f, d.vertex_C_);
  write_vec3d(f, d.vertex_attached_);
  f.write(reinterpret_cast<const char*>(&d.frame_id_), sizeof(unsigned int));
}
void STDescManager::read_stdesc(std::ifstream &f, STDesc &d) {
  read_vec3d(f, d.side_length_);
  read_vec3d(f, d.angle_);
  read_vec3d(f, d.center_);
  read_vec3d(f, d.vertex_A_);
  read_vec3d(f, d.vertex_B_);
  read_vec3d(f, d.vertex_C_);
  read_vec3d(f, d.vertex_attached_);
  f.read(reinterpret_cast<char*>(&d.frame_id_), sizeof(unsigned int));
}

// ============================================================================
//  Save / Load
// ============================================================================

bool STDescManager::Save(const std::string &dir) const
{
  namespace fs = std::filesystem;
  fs::create_directories(dir);

  // Descriptor hash table
  {
    std::ofstream f(dir + "/std_descs.bin", std::ios::binary);
    if (!f) { std::cerr << "[STD] Cannot write std_descs.bin\n"; return false; }

    uint32_t n_buckets = static_cast<uint32_t>(data_base_.size());
    f.write(reinterpret_cast<const char*>(&n_buckets), sizeof(n_buckets));

    for (const auto &kv : data_base_) {
      f.write(reinterpret_cast<const char*>(&kv.first.x), sizeof(int64_t));
      f.write(reinterpret_cast<const char*>(&kv.first.y), sizeof(int64_t));
      f.write(reinterpret_cast<const char*>(&kv.first.z), sizeof(int64_t));
      f.write(reinterpret_cast<const char*>(&kv.first.a), sizeof(int64_t));
      f.write(reinterpret_cast<const char*>(&kv.first.b), sizeof(int64_t));
      f.write(reinterpret_cast<const char*>(&kv.first.c), sizeof(int64_t));
      uint32_t n_desc = static_cast<uint32_t>(kv.second.size());
      f.write(reinterpret_cast<const char*>(&n_desc), sizeof(n_desc));
      for (const auto &d : kv.second) write_stdesc(f, d);
    }
  }

  // Plane clouds
  {
    std::ofstream f(dir + "/std_planes.bin", std::ios::binary);
    if (!f) { std::cerr << "[STD] Cannot write std_planes.bin\n"; return false; }

    uint32_t n_frames = static_cast<uint32_t>(plane_cloud_vec_.size());
    f.write(reinterpret_cast<const char*>(&n_frames), sizeof(n_frames));

    for (const auto &cloud : plane_cloud_vec_) {
      uint32_t n_pts = static_cast<uint32_t>(cloud->size());
      f.write(reinterpret_cast<const char*>(&n_pts), sizeof(n_pts));
      for (const auto &pt : cloud->points) {
        float buf[6] = {pt.x, pt.y, pt.z, pt.normal_x, pt.normal_y, pt.normal_z};
        f.write(reinterpret_cast<const char*>(buf), 6 * sizeof(float));
      }
    }
  }

  std::cout << "[STD] Saved to: " << dir
            << "  buckets=" << data_base_.size()
            << "  frames="  << plane_cloud_vec_.size() << "\n";
  return true;
}

bool STDescManager::Load(const std::string &dir)
{
  // Descriptor hash table
  {
    std::ifstream f(dir + "/std_descs.bin", std::ios::binary);
    if (!f) { std::cerr << "[STD] Cannot read: " << dir << "/std_descs.bin\n"; return false; }

    uint32_t n_buckets = 0;
    f.read(reinterpret_cast<char*>(&n_buckets), sizeof(n_buckets));
    data_base_.clear();

    for (uint32_t i = 0; i < n_buckets; i++) {
      STDesc_LOC key;
      f.read(reinterpret_cast<char*>(&key.x), sizeof(int64_t));
      f.read(reinterpret_cast<char*>(&key.y), sizeof(int64_t));
      f.read(reinterpret_cast<char*>(&key.z), sizeof(int64_t));
      f.read(reinterpret_cast<char*>(&key.a), sizeof(int64_t));
      f.read(reinterpret_cast<char*>(&key.b), sizeof(int64_t));
      f.read(reinterpret_cast<char*>(&key.c), sizeof(int64_t));
      uint32_t n_desc = 0;
      f.read(reinterpret_cast<char*>(&n_desc), sizeof(n_desc));
      std::vector<STDesc> descs(n_desc);
      for (uint32_t j = 0; j < n_desc; j++) read_stdesc(f, descs[j]);
      data_base_[key] = std::move(descs);
    }
  }

  // Plane clouds
  {
    std::ifstream f(dir + "/std_planes.bin", std::ios::binary);
    if (!f) { std::cerr << "[STD] Cannot read: " << dir << "/std_planes.bin\n"; return false; }

    uint32_t n_frames = 0;
    f.read(reinterpret_cast<char*>(&n_frames), sizeof(n_frames));
    plane_cloud_vec_.clear();
    plane_cloud_vec_.reserve(n_frames);

    for (uint32_t i = 0; i < n_frames; i++) {
      uint32_t n_pts = 0;
      f.read(reinterpret_cast<char*>(&n_pts), sizeof(n_pts));
      auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>();
      cloud->resize(n_pts);
      for (uint32_t j = 0; j < n_pts; j++) {
        float buf[6];
        f.read(reinterpret_cast<char*>(buf), 6 * sizeof(float));
        cloud->points[j].x        = buf[0];
        cloud->points[j].y        = buf[1];
        cloud->points[j].z        = buf[2];
        cloud->points[j].normal_x = buf[3];
        cloud->points[j].normal_y = buf[4];
        cloud->points[j].normal_z = buf[5];
      }
      plane_cloud_vec_.push_back(cloud);
    }

    // FIX: set current_frame_id_ so AddSTDescs increments to n_frames
    // (not n_frames+1) for the first new keyframe after loading.
    current_frame_id_ = static_cast<unsigned int>(n_frames) - 1;
  }

  std::cout << "[STD] Loaded from: " << dir
            << "  buckets=" << data_base_.size()
            << "  frames="  << plane_cloud_vec_.size() << "\n";
  return true;
}