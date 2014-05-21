/*****************************************
 * GraphicEnd
 * Read images and perform pairwise alignment, while keep
 * the constraints between keyframes.
 * Xiang 14.05
 ****************************************/
#pragma once
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

using namespace cv;
using namespace std;

struct PLANE
{
    pcl::ModelCoefficients coff;  //a,b,c,d
    vector<KeyPoint> kp;             // keypoints
    vector<Point3f> kp_pos;        // 3d position of keypoints
    Mat desp;                               // descriptor
    Mat image;                             // grayscale image with mask
};

struct KEYFRAME //关键帧: 一个关键帧由它上面的若干个平面组成
{
    int id;
    vector<PLANE> planes;
};

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class GraphicEnd
{
 public:
    GraphicEnd();
    ~GraphicEnd();

    void init();

    int run();

    int readimage();
    int process();

    void generateKeyFrame();    //将当前帧作为一个新的关键帧
    //internal
    vector<PLANE> extractPlanes( PointCloud::Ptr cloud ); //从点云提取一组平面
    void generateImageOnPlane( Mat rgb, PLANE&plane, Mat depth); //根据深度信息生成平面上的灰度图像
    vector<KeyPoint> extractKeypoints(Mat image)        
    {
        vector<KeyPoint> kp;
        _detector->detect(image, kp);
        return kp;
    }
    Mat extractDescriptor( Mat image, vector<KeyPoint>& kp) 
    {
        Mat desp;
        _descriptor->compute( image, kp, desp);
        return desp;
    }
    void compute3dPosition( PLANE& plane, Mat depth);
    vector<DMatch> match( vector<PLANE>& p1, vector<PLANE>& p2);
    vector<DMatch> match(Mat desp1, Mat desp2);

    Eigen::Isometry3d pnp( PLANE& p1, PLANE& p2 ); //求解两个平面间的PnP问题
    Eigen::Isometry3d multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2); //求解两组平面间的多PnP问题
 public:
    //data
    Eigen::Isometry3d _robot;                  //机器人的位姿，含旋转矩阵与位移向量
    vector<KEYFRAME> _keyframes;
    KEYFRAME _currKF;  //当前的关键帧
    Mat _currRGB, _currDep, _lastRGB, _lastDep; //当前帧的灰度图/深度图与上一帧的灰度/深度图
    PointCloud::Ptr _currCloud, _lastCloud;   //当前帧与上一帧点云
    Ptr<FeatureDetector> _detector;
    Ptr<DescriptorExtractor> _descriptor;
    
    //Parameters
    int _index;  //当前的图像索引
    string _rgbPath, _depPath, _pclPath;
    int _loops;
    bool _success;
    int _step_time;
    double _distance_threshold;
    double _min_error_plane;
    double _match_min_dist;
    double _percent;
};

