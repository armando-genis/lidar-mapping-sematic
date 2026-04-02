cmake_minimum_required(VERSION 3.16)
project(lidar_odom VERSION 2.0 LANGUAGES CXX C)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE STRING "Suppress developer warnings" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread -fexceptions -flto=auto")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O1 -g -pthread -fexceptions")
set(CMAKE_C_FLAGS_RELEASE "-O3 -Wall -fPIC")
add_definitions(-DROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/\")


find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PCL REQUIRED)
find_package(Glog REQUIRED)
find_package(basic REQUIRED)
find_package(GTSAM REQUIRED QUIET)


rosidl_generate_interfaces(${PROJECT_NAME}
  msg/CloudPose.msg
  msg/CloudPose2.msg
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)

include_directories(
  include
  3rdparty
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Glog_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

file(GLOB_RECURSE LIO_LIB_SRC "src/lio/*.cpp" "src/ros/*.cpp")
file(GLOB_RECURSE LIO_LIB_HEADERS "include/lio/*.hpp" "include/lio/*.h")

add_library(lio_odom ${LIO_LIB_SRC} ${LIO_LIB_HEADERS})

ament_target_dependencies(lio_odom PUBLIC
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  nav_msgs
  pcl_ros
  basic
  rosidl_default_runtime
)

target_link_libraries(lio_odom PUBLIC
  Eigen3::Eigen
  ${PCL_LIBRARIES}
  glog
  pthread
  gtsam
  ${cpp_typesupport_target}
)

add_executable(lidar_odom_node src/apps/super_lio_node.cpp)
target_link_libraries(lidar_odom_node
  lio_odom
  ${cpp_typesupport_target}
)

add_executable(lidar_odom_relocation_node src/apps/relocation_node.cpp)
target_link_libraries(lidar_odom_relocation_node
  lio_odom
  ${cpp_typesupport_target}
)

add_executable(lidar_odom_loop_node src/apps/super_lio_loop_node.cpp)
target_link_libraries(lidar_odom_loop_node
  lio_odom
  gtsam
  ${cpp_typesupport_target}
)

install(TARGETS
  lio_odom
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(TARGETS
  lidar_odom_node
  lidar_odom_loop_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  lidar_odom_relocation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  3rdparty/
  DESTINATION include/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

ament_package()
