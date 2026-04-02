#pragma once

#include <opencv2/opencv.hpp>
#include <cmath>

struct LidarRotation
{
    double axis_x;
    double axis_y;
    double axis_z;
};

class CameraLidarExtrinsics
{
public:

    CameraLidarExtrinsics(
        const cv::Mat& R_opencv_in,
        const cv::Mat& t_opencv_in,
        const cv::Mat& R_robot_in,
        const cv::Mat& t_robot_in,
        const LidarRotation& lidar_rotation_in)
    {
        lidar_rotation = lidar_rotation_in;

        cv::Mat R_lidar = euler_to_rotation(
            lidar_rotation.axis_x,
            lidar_rotation.axis_y,
            lidar_rotation.axis_z
        );

        cv::Mat R_lidar_T    = R_lidar.t();
        cv::Mat R_opencv_in_ = reshape_to_3x3(R_opencv_in);
        cv::Mat R_robot_in_  = reshape_to_3x3(R_robot_in);

        // R_opencv = R_opencv_in * R_lidar_T  (same as Python: R_base @ R_lidar.T)
        cv::gemm(R_opencv_in_, R_lidar_T, 1.0, cv::Mat(), 0.0, R_opencv);
        cv::gemm(R_robot_in_,  R_lidar_T, 1.0, cv::Mat(), 0.0, R_robot);
        t_opencv = reshape_to_3x1(t_opencv_in);
        t_robot  = reshape_to_3x1(t_robot_in);
    }

    // ── Getters ───────────────────────────────────────────────────────────────
    cv::Mat get_R_opencv() const { return R_opencv; }
    cv::Mat get_t_opencv() const { return t_opencv; }
    cv::Mat get_R_robot()  const { return R_robot; }
    cv::Mat get_t_robot()  const { return t_robot; }
    const LidarRotation& get_lidar_rotation() const { return lidar_rotation; }

    // ── Setters (used by edge calibrator to update refined extrinsics) ────────
    // Note: these set the final composed matrices directly.
    // The raw R_lidar baked-in rotation is NOT re-applied — the calibrator
    // already works in the composed (opencv) space.
    void set_R_opencv(const cv::Mat& R) { R_opencv = reshape_to_3x3(R); }
    void set_t_opencv(const cv::Mat& t) { t_opencv = reshape_to_3x1(t); }
    void set_R_robot(const cv::Mat& R)  { R_robot  = reshape_to_3x3(R); }
    void set_t_robot(const cv::Mat& t)  { t_robot  = reshape_to_3x1(t); }

    // ── Produce the "raw" R/t (before lidar rotation baked in) ───────────────
    // Useful when saving back to YAML: raw = refined * R_lidar  (inverse of constructor)
    cv::Mat get_R_opencv_raw() const
    {
        cv::Mat R_lidar = euler_to_rotation(
            lidar_rotation.axis_x, lidar_rotation.axis_y, lidar_rotation.axis_z);
        cv::Mat R_raw;
        cv::gemm(R_opencv, R_lidar, 1.0, cv::Mat(), 0.0, R_raw);
        return R_raw;
    }

    cv::Mat get_R_robot_raw() const
    {
        cv::Mat R_lidar = euler_to_rotation(
            lidar_rotation.axis_x, lidar_rotation.axis_y, lidar_rotation.axis_z);
        cv::Mat R_raw;
        cv::gemm(R_robot, R_lidar, 1.0, cv::Mat(), 0.0, R_raw);
        return R_raw;
    }

private:

    LidarRotation lidar_rotation;

    cv::Mat R_opencv;
    cv::Mat t_opencv;
    cv::Mat R_robot;
    cv::Mat t_robot;

    static cv::Mat reshape_to_3x3(const cv::Mat& mat)
    {
        cv::Mat out;
        mat.convertTo(out, CV_64F);
        return out.reshape(1, 3);
    }

    static cv::Mat reshape_to_3x1(const cv::Mat& mat)
    {
        cv::Mat out;
        mat.convertTo(out, CV_64F);
        return out.reshape(1, 3);
    }

    static cv::Mat euler_to_rotation(double ax, double ay, double az)
    {
        double cx = cos(ax), sx = sin(ax);
        double cy = cos(ay), sy = sin(ay);
        double cz = cos(az), sz = sin(az);

        cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
            1,  0,   0,
            0,  cx, -sx,
            0,  sx,  cx);

        cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
             cy, 0, sy,
              0, 1,  0,
            -sy, 0, cy);

        cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
            cz, -sz, 0,
            sz,  cz, 0,
             0,   0, 1);

        return Rz * Ry * Rx;
    }
};