#include <ros/ros.h>
#include <ros/console.h>

#include <geometry_msgs/PoseStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

//#include <jsk_recognition_msgs/BoundingBox.h>
//#include <jsk_recognition_msgs/BoundingBoxArray.h>
//#include <autoware_msgs/DetectedObjectArray.h>

#include "lidar_obstacle_detector/DetectedObjectArray.h"
#include "lidar_obstacle_detector/DetectedObject.h"
#include "lidar_obstacle_detector/Vech_state.h"
#include "lidar_obstacle_detector/Localization.h"
#include "lidar_obstacle_detector/Ins.h"
#include <visualization_msgs/MarkerArray.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <dynamic_reconfigure/server.h>
#include <lidar_obstacle_detector/obstacle_detector_Config.h>

#include "lidar_obstacle_detector/obstacle_detector.hpp"
#include "lidar_obstacle_detector/tgs.hpp"
#include "lidar_obstacle_detector/aos.hpp"
#include "lidar_obstacle_detector/patchworkplusplus.hpp"

namespace lidar_obstacle_detector 
{
float ROTATE_X, ROTATE_Y, ROTATE_Z, ROTATE_ROLL, ROTATE_PITCH, ROTATE_YAW;

// Pointcloud Filtering Parameters
bool USE_PCA_BOX;
bool USE_TRACKING;
std::string GROUND_SEGMENT_TYPE;
float VOXEL_GRID_SIZE;
Eigen::Vector4f ROI_MAX_POINT, ROI_MIN_POINT;
float GROUND_THRESH;
float CLUSTER_THRESH;
int CLUSTER_MAX_SIZE, CLUSTER_MIN_SIZE;
float DISPLACEMENT_THRESH, IOU_THRESH;

class ObstacleDetectorNode
{
 public:
  ObstacleDetectorNode();
  virtual ~ObstacleDetectorNode() {};

 private:
  Eigen::Affine3f main_transform_;
  size_t obstacle_id_;
  std::string bbox_target_frame_, bbox_source_frame_;
  std::vector<Box> prev_boxes_, curr_boxes_;
  std::shared_ptr<ObstacleDetector<pcl::PointXYZ>> obstacle_detector;
  double veh_x, veh_y, veh_z;
  float veh_course;

  ros::NodeHandle nh;
  dynamic_reconfigure::Server<lidar_obstacle_detector::obstacle_detector_Config> server;
  dynamic_reconfigure::Server<lidar_obstacle_detector::obstacle_detector_Config>::CallbackType f;

  ros::Subscriber sub_lidar_points;
  ros::Subscriber sub_odom_topic;
  ros::Publisher pub_cloud_ground;
  ros::Publisher pub_cloud_clusters;
  //ros::Publisher pub_jsk_bboxes;
  //ros::Publisher pub_autoware_objects;
  ros::Publisher pub_marker_bboxes;
  ros::Publisher pub_objects;

  void lidarPointsCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar_points);
  void odomCallback(const lidar_obstacle_detector::Localization::ConstPtr& odom);
  void publishClouds(const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr>&& segmented_clouds, const std_msgs::Header& header);
  visualization_msgs::Marker transformMarker(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed);
  lidar_obstacle_detector::DetectedObject transformObject(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed);
  //jsk_recognition_msgs::BoundingBox transformJskBbox(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed);
  //autoware_msgs::DetectedObject transformAutowareObject(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed);
  void publishDetectedObjects(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&& cloud_clusters, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&& convex_clusters, const std_msgs::Header& header);
  std::vector<geometry_msgs::Point32> calculateBoxVertices(const Eigen::Vector3f& position, const Eigen::Vector3f& dimension, const Eigen::Quaternionf& quaternion);

  boost::shared_ptr<travel::TravelGroundSeg<pcl::PointXYZ>> travel_ground_seg;
  boost::shared_ptr<travel::ObjectCluster<pcl::PointXYZ>> travel_object_seg;

  boost::shared_ptr<PatchWorkpp<pcl::PointXYZ>> PatchworkppGroundSeg;
};

// Dynamic parameter server callback function
void dynamicParamCallback(lidar_obstacle_detector::obstacle_detector_Config& config, uint32_t level)
{
  ROTATE_X = config.rotate_x;
  ROTATE_Y = config.rotate_y;
  ROTATE_Z = config.rotate_z;
  ROTATE_ROLL = config.rotate_roll;
  ROTATE_PITCH = config.rotate_pitch;
  ROTATE_YAW = config.rotate_yaw;

  // Pointcloud Filtering Parameters
  USE_PCA_BOX = config.use_pca_box;
  USE_TRACKING = config.use_tracking;
  GROUND_SEGMENT_TYPE = config.ground_segment;
  VOXEL_GRID_SIZE = config.voxel_grid_size;
  ROI_MAX_POINT = Eigen::Vector4f(config.roi_max_x, config.roi_max_y, config.roi_max_z, 1);
  ROI_MIN_POINT = Eigen::Vector4f(config.roi_min_x, config.roi_min_y, config.roi_min_z, 1);
  GROUND_THRESH = config.ground_threshold;
  CLUSTER_THRESH = config.cluster_threshold;
  CLUSTER_MAX_SIZE = config.cluster_max_size;
  CLUSTER_MIN_SIZE = config.cluster_min_size;
  DISPLACEMENT_THRESH = config.displacement_threshold;
  IOU_THRESH = config.iou_threshold;
}

ObstacleDetectorNode::ObstacleDetectorNode() : veh_x(0.0), veh_y(0.0), veh_z(0.0), veh_course(0.0)
{
  ros::NodeHandle private_nh("~");
  
  std::string lidar_points_topic;
  std::string cloud_ground_topic;
  std::string cloud_clusters_topic;
  //std::string jsk_bboxes_topic;
  //std::string autoware_objects_topic;
  std::string marker_bboxes_topic;
  std::string objects_topic;

  private_nh.param("lidar_points_topic", lidar_points_topic, std::string("/PCD_DATA0"));
  private_nh.param("cloud_ground_topic", cloud_ground_topic, std::string("obstacle_detector/cloud_ground"));
  private_nh.param("cloud_clusters_topic", cloud_clusters_topic, std::string("obstacle_detector/cloud_clusters"));
  private_nh.param("jsk_bboxes_topic", marker_bboxes_topic, std::string("obstacle_detector/jsk_bboxes"));
  private_nh.param("autoware_objects_topic", objects_topic, std::string("/autoware_tracker/cluster/objects"));
  private_nh.param("bbox_target_frame", bbox_target_frame_, std::string("/PCD_DATA0"));
  

  sub_lidar_points = nh.subscribe(lidar_points_topic, 1, &ObstacleDetectorNode::lidarPointsCallback, this);
  sub_odom_topic = nh.subscribe("/odomData", 1, &ObstacleDetectorNode::odomCallback, this);
  pub_cloud_ground = nh.advertise<sensor_msgs::PointCloud2>(cloud_ground_topic, 1);
  pub_cloud_clusters = nh.advertise<sensor_msgs::PointCloud2>(cloud_clusters_topic, 1);
  pub_marker_bboxes = nh.advertise<visualization_msgs::MarkerArray>(marker_bboxes_topic, 1);
  pub_objects = nh.advertise<lidar_obstacle_detector::DetectedObjectArray>(objects_topic, 1);
  //pub_jsk_bboxes = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>(jsk_bboxes_topic, 1);
  //pub_autoware_objects = nh.advertise<autoware_msgs::DetectedObjectArray>(autoware_objects_topic, 1);

  // Dynamic Parameter Server & Function
  f = boost::bind(&dynamicParamCallback, _1, _2);
  server.setCallback(f);

  // Create point processor
  obstacle_detector = std::make_shared<ObstacleDetector<pcl::PointXYZ>>();
  obstacle_id_ = 0;

  travel_ground_seg.reset(new travel::TravelGroundSeg<pcl::PointXYZ>(&nh));
  travel_object_seg.reset(new travel::ObjectCluster<pcl::PointXYZ>(&nh)); 
  PatchworkppGroundSeg.reset(new PatchWorkpp<pcl::PointXYZ>(&nh));
}

void ObstacleDetectorNode::odomCallback(const lidar_obstacle_detector::Localization::ConstPtr& odom )
{
  veh_x = odom->location.pose.pose.position.x;
  veh_y = odom->location.pose.pose.position.y;
  veh_z = 0;

  tf2::Quaternion quat;
  quat.setValue(
    odom->location.pose.pose.orientation.x,
    odom->location.pose.pose.orientation.y,
    odom->location.pose.pose.orientation.z,
    odom->location.pose.pose.orientation.w);
  tf2::Matrix3x3 m(quat);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  veh_course = yaw;
}

void ObstacleDetectorNode::lidarPointsCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar_points)
{
  ROS_DEBUG("lidar points recieved");
  // Time the whole process
  const auto start_time = std::chrono::steady_clock::now();
  const auto pointcloud_header = lidar_points->header;
  bbox_source_frame_ = lidar_points->header.frame_id;

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*lidar_points, *raw_cloud);
  main_transform_ = Eigen::Affine3f::Identity();
  main_transform_.rotate(Eigen::AngleAxisf(ROTATE_YAW, Eigen::Vector3f::UnitZ()));
        // Y 轴上旋转 pitch 弧度
  main_transform_.rotate(Eigen::AngleAxisf(ROTATE_PITCH, Eigen::Vector3f::UnitY()));
        // X 轴上旋转 roll 弧度
  main_transform_.rotate(Eigen::AngleAxisf(ROTATE_ROLL, Eigen::Vector3f::UnitX()));
        //在 X Y Z 轴上的平移.
  main_transform_.translation() << ROTATE_X, ROTATE_Y, ROTATE_Z;
  pcl::transformPointCloud(*raw_cloud, *raw_cloud, main_transform_);


  // Downsampleing, ROI, and removing the car roof
  auto filtered_cloud = obstacle_detector->filterCloud(raw_cloud, VOXEL_GRID_SIZE, ROI_MIN_POINT, ROI_MAX_POINT);
  
  std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmented_clouds_ptr;
  segmented_clouds_ptr.first = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  segmented_clouds_ptr.second = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

  // 根据 ground_segment 参数执行不同的操作
  if (GROUND_SEGMENT_TYPE == "RANSAC") {
    ROS_INFO("Using RANSAC for ground segmentation");
    auto segmented_clouds = obstacle_detector->segmentPlane(filtered_cloud, 30, GROUND_THRESH);
    auto cloudNonground_out_ptr = travel_object_seg->segmentObjects(segmented_clouds.first);
    segmented_clouds_ptr.first = cloudNonground_out_ptr;
    segmented_clouds_ptr.second = segmented_clouds.second;

    // 调用 RANSAC 相关的函数
  } else if (GROUND_SEGMENT_TYPE == "RGPF") {
    ROS_INFO("Using RGPF for ground segmentation");
    auto segmented_clouds = PatchworkppGroundSeg->estimate_ground(*filtered_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNonground_ptr(new pcl::PointCloud<pcl::PointXYZ>(segmented_clouds.first));
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNonground_out_ptr(new pcl::PointCloud<pcl::PointXYZ>(segmented_clouds.first));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudGround_out_ptr(new pcl::PointCloud<pcl::PointXYZ>(segmented_clouds.second));
    auto cloudNonground_out_ptr = travel_object_seg->segmentObjects(cloudNonground_ptr);
    segmented_clouds_ptr.first = cloudNonground_out_ptr;
    segmented_clouds_ptr.second = cloudGround_out_ptr;

    // 调用 RGPF 相关的函数
  } else if (GROUND_SEGMENT_TYPE == "TRAVEL") {
    ROS_INFO("Using TRAVEL for ground segmentation");
    // Apply traversable ground segmentation
    auto segmented_clouds = travel_ground_seg->estimateGround(*filtered_cloud);
    // 创建指向 PointCloud 的智能指针
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNonground_ptr(new pcl::PointCloud<pcl::PointXYZ>(segmented_clouds.first));
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNonground_out_ptr(new pcl::PointCloud<pcl::PointXYZ>(segmented_clouds.first));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudGround_out_ptr(new pcl::PointCloud<pcl::PointXYZ>(segmented_clouds.second));
    auto cloudNonground_out_ptr = travel_object_seg->segmentObjects(cloudNonground_ptr);
    segmented_clouds_ptr.first = cloudNonground_out_ptr;
    segmented_clouds_ptr.second = cloudGround_out_ptr;
    // 调用 TRAVEL 相关的函数
  } else {
    ROS_WARN("Unknown ground segment type: %s", GROUND_SEGMENT_TYPE.c_str());
  }
  // Cluster objects
  auto cloud_clusters = obstacle_detector->clustering(segmented_clouds_ptr.first, CLUSTER_THRESH, CLUSTER_MIN_SIZE, CLUSTER_MAX_SIZE);
  // Convex hull objects
  auto convex_clusters = obstacle_detector->computeConvexHulls(cloud_clusters);

  // Publish ground cloud and obstacle cloud
  publishClouds(std::move(segmented_clouds_ptr), pointcloud_header);
  // Publish Obstacles
  publishDetectedObjects(std::move(cloud_clusters), std::move(convex_clusters), pointcloud_header);

  // Time the whole process
  const auto end_time = std::chrono::steady_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  ROS_INFO("The obstacle_detector_node found %d obstacles in %.3f second", int(prev_boxes_.size()), float(elapsed_time.count()/1000.0));
}

void ObstacleDetectorNode::publishClouds(const std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr>&& segmented_clouds, const std_msgs::Header& header)
{
  sensor_msgs::PointCloud2::Ptr ground_cloud(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*(segmented_clouds.second), *ground_cloud);
  ground_cloud->header = header;

  sensor_msgs::PointCloud2::Ptr obstacle_cloud(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*(segmented_clouds.first), *obstacle_cloud);
  obstacle_cloud->header = header;

  pub_cloud_ground.publish(std::move(ground_cloud));
  pub_cloud_clusters.publish(std::move(obstacle_cloud));
}

visualization_msgs::Marker ObstacleDetectorNode::transformMarker(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed)
{
  visualization_msgs::Marker marker_bbox;
  marker_bbox.header = header;
  marker_bbox.pose = pose_transformed;
  marker_bbox.type = visualization_msgs::Marker::LINE_STRIP;
  marker_bbox.id = box.id;
  geometry_msgs::Point pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8;
  pos1.x = box.dimension(0) / 2;
  pos1.y = box.dimension(1) / 2;
  pos1.z = box.dimension(2) / 2;

  pos2.x = box.dimension(0) / 2;
  pos2.y = box.dimension(1) / 2;
  pos2.z = -box.dimension(2) / 2;

  pos3.x = box.dimension(0) / 2;
  pos3.y = -box.dimension(1) / 2;
  pos3.z = -box.dimension(2) / 2;

  pos4.x = box.dimension(0) / 2;
  pos4.y = -box.dimension(1) / 2;
  pos4.z = box.dimension(2) / 2;

  pos5.x = -box.dimension(0) / 2;
  pos5.y = -box.dimension(1) / 2;
  pos5.z = box.dimension(2) / 2;

  pos6.x = -box.dimension(0) / 2;
  pos6.y = -box.dimension(1) / 2;
  pos6.z = -box.dimension(2) / 2;

  pos7.x = -box.dimension(0) / 2;
  pos7.y = box.dimension(1) / 2;
  pos7.z = -box.dimension(2) / 2;

  pos8.x = -box.dimension(0) / 2;
  pos8.y = box.dimension(1) / 2;
  pos8.z = box.dimension(2) / 2;
  marker_bbox.points.push_back(pos1);
  marker_bbox.points.push_back(pos2);
  marker_bbox.points.push_back(pos3);
  marker_bbox.points.push_back(pos4);
  marker_bbox.points.push_back(pos5);
  marker_bbox.points.push_back(pos6);
  marker_bbox.points.push_back(pos7);
  marker_bbox.points.push_back(pos8);
  marker_bbox.points.push_back(pos1);
  marker_bbox.points.push_back(pos4);
  marker_bbox.points.push_back(pos3);
  marker_bbox.points.push_back(pos6);
  marker_bbox.points.push_back(pos5);
  marker_bbox.points.push_back(pos8);
  marker_bbox.points.push_back(pos7);
  marker_bbox.points.push_back(pos2);
  marker_bbox.color.r = 1.0;
  marker_bbox.color.g = 0.0;
  marker_bbox.color.b = 0.0;
  marker_bbox.scale.x = 0.1;
  marker_bbox.color.a = 1.0;
  marker_bbox.lifetime.fromSec(0.1);
  return std::move(marker_bbox);
}

/* jsk_recognition_msgs::BoundingBox ObstacleDetectorNode::transformJskBbox(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed)
{
  jsk_recognition_msgs::BoundingBox jsk_bbox;
  jsk_bbox.header = header;
  jsk_bbox.pose = pose_transformed;
  jsk_bbox.dimensions.x = box.dimension(0);
  jsk_bbox.dimensions.y = box.dimension(1);
  jsk_bbox.dimensions.z = box.dimension(2);
  jsk_bbox.value = 1.0f;
  jsk_bbox.label = box.id;

  return std::move(jsk_bbox);
} */

lidar_obstacle_detector::DetectedObject ObstacleDetectorNode::transformObject(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed)
{
  lidar_obstacle_detector::DetectedObject autoware_object;
  autoware_object.header = header;
  autoware_object.id = box.id;
  autoware_object.label = "unknown";
  autoware_object.score = 1.0f;
  autoware_object.pose = pose_transformed;
  autoware_object.pose_reliable = true;
  autoware_object.dimensions.x = box.dimension(0);
  autoware_object.dimensions.y = box.dimension(1);
  autoware_object.dimensions.z = box.dimension(2);
  autoware_object.valid = true;
  autoware_object.convex_hull.polygon.points = box.convex_hull;
  return std::move(autoware_object);
}
/* autoware_msgs::DetectedObject ObstacleDetectorNode::transformAutowareObject(const Box& box, const std_msgs::Header& header, const geometry_msgs::Pose& pose_transformed)
{
  autoware_msgs::DetectedObject autoware_object;
  autoware_object.header = header;
  autoware_object.id = box.id;
  autoware_object.label = "unknown";
  autoware_object.score = 1.0f;
  autoware_object.pose = pose_transformed;
  autoware_object.pose_reliable = true;
  autoware_object.dimensions.x = box.dimension(0);
  autoware_object.dimensions.y = box.dimension(1);
  autoware_object.dimensions.z = box.dimension(2);
  autoware_object.valid = true;

  return std::move(autoware_object);
} */

void ObstacleDetectorNode::publishDetectedObjects(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&& cloud_clusters, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>&& convex_clusters, const std_msgs::Header& header)
{
  for (size_t i = 0; i < cloud_clusters.size(); ++i)
  {
    auto& cluster = cloud_clusters[i];
    auto& convex_cluster = convex_clusters[i];

    // Create Bounding Box
    Box box = USE_PCA_BOX ? 
        obstacle_detector->pcaBoundingBox(cluster, obstacle_id_) : 
        obstacle_detector->axisAlignedBoundingBox(cluster, obstacle_id_);

    // 为 Box 赋值凸包
    std::vector<geometry_msgs::Point32> convex_hull_points = calculateBoxVertices(box.position, box.dimension, box.quaternion);
    for (const auto& point : convex_cluster->points)
    {
        geometry_msgs::Point32 convex_point;
        convex_point.x = point.x;
        convex_point.y = point.y;
        convex_point.z = box.position[2]-box.dimension[2]/2;  // 障碍物底面
        convex_hull_points.push_back(convex_point);
    }
    convex_hull_points.push_back(convex_hull_points[8]);//形成闭环

    // 通过新的构造函数创建 Box，并赋值 convex_hull
    box = Box(obstacle_id_, box.position, box.dimension, box.quaternion, convex_hull_points);

    obstacle_id_ = (obstacle_id_ < SIZE_MAX) ? ++obstacle_id_ : 0;
    curr_boxes_.emplace_back(box);
  }
  // Re-assign Box ids based on tracking result
  if (USE_TRACKING)
    obstacle_detector->obstacleTracking(prev_boxes_, curr_boxes_, DISPLACEMENT_THRESH, IOU_THRESH);
  
  // Lookup for frame transform between the lidar frame and the target frame
  auto bbox_header = header;
  bbox_header.frame_id = bbox_target_frame_;
  // Construct Bounding Boxes from the clusters
  visualization_msgs::MarkerArray jsk_bboxes;
  lidar_obstacle_detector::DetectedObjectArray autoware_objects;
  autoware_objects.header = bbox_header;
  autoware_objects.vech_st.veh_x = veh_x;
  autoware_objects.vech_st.veh_y = veh_y;
  autoware_objects.vech_st.veh_z = veh_z;
  autoware_objects.vech_st.veh_Course = veh_course;
  //jsk_recognition_msgs::BoundingBoxArray jsk_bboxes;
  //jsk_bboxes.header = bbox_header;
  //autoware_msgs::DetectedObjectArray autoware_objects;
  //autoware_objects.header = bbox_header;

  // Transform boxes from lidar frame to base_link frame, and convert to jsk and autoware msg formats
  for (auto& box : curr_boxes_)
  {
    geometry_msgs::Pose pose;
    pose.position.x = box.position(0);
    pose.position.y = box.position(1);
    pose.position.z = box.position(2);
    pose.orientation.w = box.quaternion.w();
    pose.orientation.x = box.quaternion.x();
    pose.orientation.y = box.quaternion.y();
    pose.orientation.z = box.quaternion.z();

    jsk_bboxes.markers.emplace_back(transformMarker(box, bbox_header, pose));
    //jsk_bboxes.boxes.emplace_back(transformJskBbox(box, bbox_header, pose_transformed));
    autoware_objects.objects.emplace_back(transformObject(box, bbox_header, pose));
    //autoware_objects.objects.emplace_back(transformAutowareObject(box, bbox_header, pose_transformed));
  }
  pub_marker_bboxes.publish(std::move(jsk_bboxes));
  //pub_jsk_bboxes.publish(std::move(jsk_bboxes));
  pub_objects.publish(std::move(autoware_objects));
  //pub_autoware_objects.publish(std::move(autoware_objects));

  // Update previous bounding boxes
  prev_boxes_.swap(curr_boxes_);
  curr_boxes_.clear();
}

// 计算Box的8个顶点
std::vector<geometry_msgs::Point32> ObstacleDetectorNode::calculateBoxVertices(const Eigen::Vector3f& position, 
                                          const Eigen::Vector3f& dimension, 
                                          const Eigen::Quaternionf& quaternion)
{
    // Box的长宽高（深度、宽度、高度）
    float length = dimension[0];  // 深度
    float width = dimension[1];   // 宽度
    float height = dimension[2];  // 高度

    // 定义盒子在局部坐标系中的8个顶点的相对位置
    Eigen::Vector3f local_corners[8] = {
        // 底面四个点
        Eigen::Vector3f(-length / 2, -width / 2, -height / 2),
        Eigen::Vector3f(-length / 2,  width / 2, -height / 2),
        Eigen::Vector3f( length / 2,  width / 2, -height / 2),
        Eigen::Vector3f( length / 2, -width / 2, -height / 2),
        // 顶面四个点
        Eigen::Vector3f(-length / 2, -width / 2,  height / 2),
        Eigen::Vector3f(-length / 2,  width / 2,  height / 2),
        Eigen::Vector3f( length / 2,  width / 2,  height / 2),
        Eigen::Vector3f( length / 2, -width / 2,  height / 2)
    };

    // 结果顶点存储
    std::vector<geometry_msgs::Point32> global_corners;

    // 旋转 + 平移
    for (int i = 0; i < 8; ++i)
    {
        // 旋转顶点
        Eigen::Vector3f rotated_corner = quaternion * local_corners[i];
        
        // 将旋转后的顶点平移到全局坐标系
        Eigen::Vector3f global_corner = rotated_corner + position;

        geometry_msgs::Point32 corner;

        corner.x = global_corner[0];
        corner.y = global_corner[1];
        corner.z = global_corner[2];

        // 存储结果
        global_corners.push_back(corner);
    }

    return global_corners;
}


} // namespace lidar_obstacle_detector

int main(int argc, char** argv)
{
  ros::init(argc, argv, "obstacle_detector_node");
  lidar_obstacle_detector::ObstacleDetectorNode obstacle_detector_node;
  ros::spin();
  return 0;
}