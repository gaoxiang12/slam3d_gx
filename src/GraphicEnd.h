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
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>

using namespace g2o;
using namespace cv;
using namespace std;

class GraphicEnd;
class SLAMEnd;

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

    void init(SLAMEnd* _pSLAMEnd);

    int run();

    int readimage();
    int process();

    void generateKeyFrame();    //将当前帧作为一个新的关键帧
    //internal

    vector<PLANE> extractPlanes( PointCloud::Ptr cloud ); //从点云提取一组平面
    
    void generateImageOnPlane( Mat rgb, PLANE&plane, Mat depth); //根据深度信息生成平面上的灰度图像

    //提取关键点
    vector<KeyPoint> extractKeypoints(Mat image)        
    {
        vector<KeyPoint> kp;
        _detector->detect(image, kp);
        return kp;
    }

    //提取描述子
    Mat extractDescriptor( Mat image, vector<KeyPoint>& kp) 
    {
        Mat desp;
        _descriptor->compute( image, kp, desp);
        return desp;
    }

    //确定平面上的关键点在空间的坐标
    void compute3dPosition( PLANE& plane, Mat depth);

    //匹配两组平面，以法向量为特征
    vector<DMatch> match( vector<PLANE>& p1, vector<PLANE>& p2);

    //匹配两组特征
    vector<DMatch> match(Mat desp1, Mat desp2);

    //求解两个平面间的PnP问题
    Eigen::Isometry3d pnp( PLANE& p1, PLANE& p2 ); 

    //求解两组平面间的多PnP问题，算法将调用SLAM端构造局部子图
    Eigen::Isometry3d multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2);
    
 public:
    //data
    SLAMEnd* _pSLAMEnd;
    
    Eigen::Isometry3d _robot;                  //机器人的位姿，含旋转矩阵与位移向量
    vector<KEYFRAME> _keyframes;      //过去的关键帧
    KEYFRAME _currKF;                              //当前的关键帧
    KEYFRAME _present;                            //当前帧，也就是正在处理的那一帧
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
    double _max_pos_change;
    stringstream ss;
};

/* ****************************************
 * SLAM End
 * 求解SLAM问题的后端
 * 提供全局求解与帧间求解两个函数
 ****************************************/
typedef BlockSolver< BlockSolverTraits<-1, -1> >  SlamBlockSolver;
typedef LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

class SLAMEnd
{
 public:
    SLAMEnd() {
        
    }
    ~SLAMEnd() {
        
    }

    void init( GraphicEnd* p)
    {
        _pGraphicEnd = p;
        SlamLinearSolver* linearSolver = new SlamLinearSolver();
        linearSolver->setBlockOrdering( false );
        SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
        solver = new OptimizationAlgorithmGaussNewton( blockSolver );
        _robustKernel = RobustKernelFactory::instance()->construct( "Cauchy" );
        globalOptimizer.setAlgorithm( solver );
        
        setupLocalOptimizer();
    }

    void setupLocalOptimizer();
        

 public:
    GraphicEnd* _pGraphicEnd;
    SparseOptimizer globalOptimizer;
    OptimizationAlgorithmGaussNewton* solver;
    RobustKernel* _robustKernel;
};
