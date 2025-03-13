# lidar_obstacle_detector

一个ros下3D lidar障碍物检测功能包，包括多种地面分割方法（RANSAC,PatchWork++,Travel）以及障碍物聚类，匈牙利匹配算法和UKF实现跟踪</br>
主体框架改编自 https://github.com/SS47816/lidar_obstacle_detector </br>
PatchWork++实现来源自 https://github.com/url-kaist/patchwork-plusplus-ros </br>
Travel实现来源自 https://github.com/url-kaist/TRAVEL</br>

## Step1
```bash
catkin_make
source devel/setup.bash
```
## Step2
在`launch\run.launch`里修改你的lidar话题名称
|参数名称|功能描述|备注|
|---|---|---|
|lidar_points_topic|lidar的话题名称| - |
|cloud_ground_topic|发布的地面点云话题名称| - |
|cloud_clusters_topic|发布的障碍物点云话题名称| - |
|jsk_bboxes_topic|障碍物boudingbox可视化话题名称| - |
|autoware_objects_topic|障碍物话题名称| - |
|bbox_target_frame|发布消息的frame_id| - |

## Step3
运行下述命令启动节点
```bash
roslaunch lidar_obstacle_detector run.launch
```
在rqt_reconfig界面可以动态调整除PatchWork++与Travel外的参数，PatchWork++与Travel的参数调整在`cfg\rslidar.yaml`下