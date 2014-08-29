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
#include <g2o/core/optimization_algorithm_levenberg.h>

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
    int frame_index;
    vector<PLANE> planes;
};

struct RESULT_OF_MULTIPNP
{
    RESULT_OF_MULTIPNP() {
        T = Eigen::Isometry3d::Identity();
        norm = 0.0;
        inliers = 0;
    }
    Eigen::Isometry3d T;
    double norm;
    int inliers;
};

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class GraphicEnd
{
 public:
    GraphicEnd();
    ~GraphicEnd();

    virtual void init(SLAMEnd* _pSLAMEnd);

    virtual int run();

    virtual int readimage();
    int process();

    //将present作为一个新的关键帧，传入current到present的变换矩阵
    virtual void generateKeyFrame( Eigen::Isometry3d T );

    //输出最后点云结果
    virtual void saveFinalResult( string fileaddr );
    //internal

    vector<PLANE> extractPlanes( PointCloud::Ptr cloud ); //从点云提取一组平面
    
    void generateImageOnPlane( Mat rgb, vector<PLANE>& planes, Mat depth); //根据深度信息生成平面上的灰度图像

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
    virtual void compute3dPosition( PLANE& plane, Mat depth);

    //匹配两组平面，以法向量为特征
    vector<DMatch> match( vector<PLANE>& p1, vector<PLANE>& p2);

    //匹配两组特征
    vector<DMatch> match(Mat desp1, Mat desp2);

    //求解两个平面间的PnP问题
    vector<DMatch> pnp( PLANE& p1, PLANE& p2 ); 

    //求解两组平面间的多PnP问题，算法将调用SLAM端构造局部子图
    virtual RESULT_OF_MULTIPNP multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2, bool loopclosure = false, int frame_index = 0, int minimum_inliers = 12);

    //闭环检测
    void loopClosure();
    void displayLC(int frame1, int frame2, double norm); //显示检测到的闭环

    //关键帧检测
    int testKeyframe();   //检测关键帧的质量
    // 丢失恢复
    void lostRecovery();

    //读取里程计
    Eigen::Isometry3d readOdometry(ifstream& fin)
    {
        string nothing[ 5 ];
        for (int i=0; i<5; i++)
            fin>>nothing[ i ];
        double data[ 7 ];
        for (int i=0; i<7; i++)
        {
            fin>>data[ i ];
        }
        fin.ignore();
        g2o::VertexSE3 v;
        v.setEstimateData( data );
        Eigen::Vector3d rpy = v.estimate().rotation().eulerAngles(2, 0, 2);
        Eigen::Isometry3d T;
        //Eigen::Translation<double, 3> trans(-data[1], -data[2], data[0]);
        Eigen::AngleAxisd r(rpy[2], -Eigen::Vector3d::UnitY());
        T = r;
        T.matrix()(0,3) = -data[1];
        T.matrix()(1,3) = -data[2];
        T.matrix()(2,3) = data[0];
        return T;
    }
 public:
    //data
    SLAMEnd* _pSLAMEnd;
    
    Eigen::Isometry3d _robot;                  //机器人的位姿，含旋转矩阵与位移向量
    Eigen::Isometry3d _kf_pos;                //机器人在关键帧上的位姿

    Eigen::Isometry3d _odo_last, _odo_this;
    Eigen::Isometry3d _robot2camera;    //机器人坐标系到相机坐标系的变换
    vector<KEYFRAME> _keyframes;      //过去的关键帧
    KEYFRAME _currKF;                              //当前的关键帧
    KEYFRAME _present;                            //当前帧，也就是正在处理的那一帧
    Mat _currRGB, _currDep, _lastRGB, _lastDep; //当前帧的灰度图/深度图与上一帧的灰度/深度图
    PointCloud::Ptr _currCloud, _lastCloud;   //当前帧与上一帧点云

    int _lost; //丢失的帧数
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
    double _error_threshold;
    int _max_planes;
    bool _loop_closure_detection;
    int _loopclosure_frames;
    double _loop_closure_error;
    int _lost_frames;
    stringstream ss;

    vector<int> _seed; //Loop Closure 种子关键帧
    bool _use_odometry;
    double _error_odometry;
    vector<Eigen::Isometry3d> _odometry;
};

/* ****************************************
 * SLAM End
 * 求解SLAM问题的后端
 ****************************************/
//typedef BlockSolver< BlockSolverTraits<-1, -1> >  SlamBlockSolver;
typedef BlockSolver_6_3 SlamBlockSolver;
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
        
        //solver = new OptimizationAlgorithmGaussNewton( blockSolver );
        solver = new OptimizationAlgorithmLevenberg( blockSolver );
        _robustKernel = RobustKernelFactory::instance()->construct( "Cauchy" );
        globalOptimizer.setVerbose( false );
        globalOptimizer.setAlgorithm( solver );
    }


 public:
    GraphicEnd* _pGraphicEnd;
    SparseOptimizer globalOptimizer;
    OptimizationAlgorithmLevenberg* solver;
    RobustKernel* _robustKernel;
};

/*****************************************
 * Graphic End 2: only use image features without planes
 ****************************************/

class GraphicEnd2: public GraphicEnd
{
 public:
    GraphicEnd2();
    ~GraphicEnd2();

 public:
    virtual int readimage();
    virtual void init( SLAMEnd* pSLAMEnd);
    virtual int run();
    void compute3dPosition( PLANE& plane, Mat depth);
    PLANE extractKPandDesp( Mat& rgb, Mat& dep);
    virtual RESULT_OF_MULTIPNP multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2, bool loopclosure = false, int frame_index = 0, int minimum_inliers = 12);
};
