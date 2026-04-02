#pragma once

#include <filesystem>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <regex>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "CameraLidarExtrinsics.hpp"
#include "CameraUndistorter.hpp"

namespace fs = std::filesystem;

class CalibrationLoader
{
public:

    std::vector<std::shared_ptr<CameraUndistorter>>      camera_array;
    std::vector<std::shared_ptr<CameraLidarExtrinsics>>  extrinsics_array;

    // ── Load ──────────────────────────────────────────────────────────────────
    void load(const std::string& calib_dir)
    {
        std::regex cam_intrinsic_re("cam(\\d+)_intrinsic.yaml");
        std::regex lidar_cam_re("LidartoCam(\\d+).yaml");

        std::map<int, std::string> intrinsic_files;
        std::map<int, std::string> extrinsic_files;

        for (const auto& entry : fs::directory_iterator(calib_dir))
        {
            std::string name = entry.path().filename();
            std::smatch match;

            if (std::regex_match(name, match, cam_intrinsic_re))
                intrinsic_files[std::stoi(match[1])] = entry.path();

            if (std::regex_match(name, match, lidar_cam_re))
                extrinsic_files[std::stoi(match[1])] = entry.path();
        }

        LidarRotation lidar_rotation = load_lidar_rotation(calib_dir);

        int max_cam = 0;
        if (!intrinsic_files.empty())
            max_cam = std::max(max_cam, intrinsic_files.rbegin()->first);
        if (!extrinsic_files.empty())
            max_cam = std::max(max_cam, extrinsic_files.rbegin()->first);

        camera_array.resize(max_cam + 1);
        extrinsics_array.resize(max_cam + 1);

        for (int cam = 0; cam <= max_cam; cam++)
        {
            if (intrinsic_files.count(cam))
                camera_array[cam] = load_intrinsics(intrinsic_files[cam]);

            if (extrinsic_files.count(cam))
                extrinsics_array[cam] = load_extrinsics(extrinsic_files[cam], lidar_rotation);
        }

        // Remember file paths so save() knows where to write
        extrinsic_paths_ = extrinsic_files;
        calib_dir_       = calib_dir;

        std::cout << "Calibration loaded successfully from " << calib_dir << "\n";
    }

    // ── Save refined extrinsics back to the original YAML files ──────────────
    // Only the R and t values are updated; all other YAML structure is preserved.
    void save(const std::string& calib_dir)
    {
        for (auto& [cam_idx, path] : extrinsic_paths_)
        {
            if (static_cast<size_t>(cam_idx) >= extrinsics_array.size()) continue;
            auto ext = extrinsics_array[cam_idx];
            if (!ext) continue;

            // Load existing YAML so we preserve every other field
            YAML::Node root;
            try { root = YAML::LoadFile(path); }
            catch (const std::exception& e)
            {
                std::cerr << "[CalibrationLoader] Cannot read " << path
                          << ": " << e.what() << "\n";
                continue;
            }

            // Write refined R and t into the opencv_frame node
            // We store the "raw" values (before lidar rotation is baked in)
            // so the next load() produces the same composed matrix.
            cv::Mat R_raw = ext->get_R_opencv_raw();
            cv::Mat t     = ext->get_t_opencv();

            root["extrinsics"]["opencv_frame"]["R"]["data"] = mat_to_yaml(R_raw);
            root["extrinsics"]["opencv_frame"]["t"]["data"] = vec_to_yaml(t);

            // Also update robot_frame R
            cv::Mat R_robot_raw = ext->get_R_robot_raw();
            cv::Mat t_robot     = ext->get_t_robot();
            root["extrinsics"]["robot_frame"]["R"]["data"] = mat_to_yaml(R_robot_raw);
            root["extrinsics"]["robot_frame"]["t"]["data"] = vec_to_yaml(t_robot);

            // Write back to file
            std::ofstream fout(path);
            if (!fout.is_open())
            {
                std::cerr << "[CalibrationLoader] Cannot write " << path << "\n";
                continue;
            }
            fout << root;
            fout.close();

            std::cout << "[CalibrationLoader] Saved refined extrinsics for cam "
                      << cam_idx << " → " << path << "\n";
        }
    }

private:

    std::string              calib_dir_;
    std::map<int, std::string> extrinsic_paths_;  // remembered at load time

    // ── YAML helpers ──────────────────────────────────────────────────────────
    static YAML::Node mat_to_yaml(const cv::Mat& M)
    {
        // M must be 3×3 CV_64F; emit row-major flat list
        YAML::Node node;
        for (int r = 0; r < M.rows; ++r)
            for (int c = 0; c < M.cols; ++c)
                node.push_back(M.at<double>(r, c));
        return node;
    }

    static YAML::Node vec_to_yaml(const cv::Mat& v)
    {
        // v must be 3×1 CV_64F
        YAML::Node node;
        for (int r = 0; r < v.rows; ++r)
            node.push_back(v.at<double>(r, 0));
        return node;
    }

    // ── Loaders (unchanged from original) ─────────────────────────────────────
    std::shared_ptr<CameraUndistorter> load_intrinsics(const std::string& path)
    {
        YAML::Node data = YAML::LoadFile(path);

        int w = data["image_width"].as<int>();
        int h = data["image_height"].as<int>();

        std::vector<double> Kvec = data["camera_matrix"]["data"].as<std::vector<double>>();
        std::vector<double> Dvec = data["distortion_coefficients"]["data"].as<std::vector<double>>();

        cv::Mat K(3, 3, CV_64F, Kvec.data());
        cv::Mat D(static_cast<int>(Dvec.size()), 1, CV_64F, Dvec.data());

        return std::make_shared<CameraUndistorter>(K.clone(), D.clone(), cv::Size(w, h));
    }

    LidarRotation load_lidar_rotation(const std::string& calib_dir)
    {
        LidarRotation rot{0.0, 0.0, 0.0};
        std::string lidar_config_path = calib_dir + "/lidarConfig.yaml";

        if (!fs::exists(lidar_config_path))
        {
            std::cout << "lidarConfig.yaml not found at " << lidar_config_path
                      << ". Using lidar_rotation 0,0,0.\n";
            return rot;
        }

        try
        {
            YAML::Node data = YAML::LoadFile(lidar_config_path);
            if (data["lidar_rotation"])
            {
                auto lr    = data["lidar_rotation"];
                rot.axis_x = lr["axis_x"].as<double>(0.0);
                rot.axis_y = lr["axis_y"].as<double>(0.0);
                rot.axis_z = lr["axis_z"].as<double>(0.0);
            }
            std::cout << "Loaded lidar_rotation: axis_x=" << rot.axis_x
                      << " axis_y=" << rot.axis_y << " axis_z=" << rot.axis_z << "\n";
        }
        catch (const std::exception& e)
        {
            std::cout << "Failed to load lidarConfig.yaml: " << e.what()
                      << ". Using 0,0,0.\n";
        }
        return rot;
    }

    std::shared_ptr<CameraLidarExtrinsics> load_extrinsics(
        const std::string& path,
        const LidarRotation& lidar_rotation)
    {
        YAML::Node data = YAML::LoadFile(path);

        auto opencv = data["extrinsics"]["opencv_frame"];
        auto robot  = data["extrinsics"]["robot_frame"];

        cv::Mat R_opencv = mat3(opencv["R"]);
        cv::Mat t_opencv = vec3(opencv["t"]);
        cv::Mat R_robot  = mat3(robot["R"]);
        cv::Mat t_robot  = vec3(robot["t"]);

        return std::make_shared<CameraLidarExtrinsics>(
            R_opencv, t_opencv, R_robot, t_robot, lidar_rotation);
    }

    cv::Mat mat3(const YAML::Node& node)
    {
        std::vector<double> v = node["data"].as<std::vector<double>>();
        return cv::Mat(3, 3, CV_64F, v.data()).clone();
    }

    cv::Mat vec3(const YAML::Node& node)
    {
        std::vector<double> v = node["data"].as<std::vector<double>>();
        return cv::Mat(3, 1, CV_64F, v.data()).clone();
    }
};