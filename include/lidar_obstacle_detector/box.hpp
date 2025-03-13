#pragma once

#include <Eigen/Geometry> 
#include <geometry_msgs/Point32.h>
struct Box
{
 public:
    int id;
    Eigen::Vector3f position;
    Eigen::Vector3f dimension;
    Eigen::Quaternionf quaternion;
    
    // 添加字段用于存储凸包点
    std::vector<geometry_msgs::Point32> convex_hull;

    // 默认构造函数
    Box() {};

    // 带位置和尺寸的构造函数，默认四元数为单位四元数
    Box(int id, Eigen::Vector3f position, Eigen::Vector3f dimension)
        : id(id), position(position), dimension(dimension), quaternion(Eigen::Quaternionf(1, 0, 0, 0))
    {}

    // 带四元数的完整构造函数
    Box(int id, Eigen::Vector3f position, Eigen::Vector3f dimension, Eigen::Quaternionf quaternion)
        : id(id), position(position), dimension(dimension), quaternion(quaternion)
    {}

    // 新的构造函数，带凸包信息
    Box(int id, Eigen::Vector3f position, Eigen::Vector3f dimension, Eigen::Quaternionf quaternion, 
        const std::vector<geometry_msgs::Point32>& convex_hull)
        : id(id), position(position), dimension(dimension), quaternion(quaternion), convex_hull(convex_hull)
    {}
};
