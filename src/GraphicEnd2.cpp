/*****************************************
 * GraphicEnd2: implementation of GraphicEnd2
 ****************************************/
#include "GraphicEnd.h"
#include "ParameterReader.h"
#include "const.h"
//CV
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

//Std
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <algorithm>

//PCL
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
//G2O
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>

using namespace std;
using namespace cv;
using namespace g2o;

GraphicEnd2::GraphicEnd2()
{
    g_pParaReader = new ParameterReader( parameter_file_addr );
    initModule_nonfree();
    _detector = FeatureDetector::create( g_pParaReader->GetPara("detector_name") );
    _descriptor = DescriptorExtractor::create( g_pParaReader->GetPara("descriptor_name") );
    _robot = Isometry3d::Identity();
    srand( (unsigned int) time(0) );
}

GraphicEnd2::~GraphicEnd2()
{
    delete g_pParaReader;
}

int GraphicEnd2::readimage()
{
    ss<<_rgbPath<<_index<<".png";
    _currRGB = imread(ss.str(), 0);
    ss.str("");
    ss.clear();

    ss<<_depPath<<_index<<".png";
    _currDep = imread(ss.str(), -1);
    ss.str("");
    ss.clear();
    
}

void GraphicEnd2::init( SLAMEnd* pSLAMEnd)
{
    cout<<"Graphic end 2 init.."<<endl;
    
    _pSLAMEnd = pSLAMEnd;
    _index = atoi( g_pParaReader->GetPara("start_index").c_str() );
    _rgbPath = g_pParaReader->GetPara("data_source")+string("/rgb_index/");
    _depPath = g_pParaReader->GetPara("data_source")+string("/dep_index/");
    _pclPath = g_pParaReader->GetPara("data_source")+string("/pcd/");
    _loops = 0;
    _success = false;
    _step_time = atoi(g_pParaReader->GetPara("step_time").c_str());
    _distance_threshold = atof( g_pParaReader->GetPara("distance_threshold").c_str() );
    _error_threshold = atof( g_pParaReader->GetPara("error_threshold").c_str() );
    _min_error_plane = atof( g_pParaReader->GetPara("min_error_plane").c_str() );
    _match_min_dist = atof( g_pParaReader->GetPara("match_min_dist").c_str() );
    _percent = atof( g_pParaReader->GetPara("plane_percent").c_str() );
    _max_pos_change = atof( g_pParaReader->GetPara("max_pos_change").c_str());
    _max_planes = atoi( g_pParaReader->GetPara("max_planes").c_str() );
    _loopclosure_frames = atoi( g_pParaReader->GetPara("loopclosure_frames").c_str() );
    _loop_closure_detection = (g_pParaReader->GetPara("loop_closure_detection") == string("yes"))?true:false;
    _loop_closure_error = atof(g_pParaReader->GetPara("loop_closure_error").c_str());
    _lost_frames = atoi( g_pParaReader->GetPara("lost_frames").c_str() );
    _robot = _kf_pos = Eigen::Isometry3d::Identity();
    _use_odometry = g_pParaReader->GetPara("use_odometry") == string("yes");
    _error_odometry = atof( g_pParaReader->GetPara("error_odometry").c_str() );
    _robot2camera = AngleAxisd(-0.5*M_PI, Vector3d::UnitY()) * AngleAxisd(0.5*M_PI, Vector3d::UnitX());
    if (_use_odometry)
    {
        cout<<"using odometry"<<endl;
        string fileaddr = g_pParaReader->GetPara("data_source")+string("/associate.txt");
        ifstream fin(fileaddr.c_str());
        if (!fin)
        {
            cerr<<"cannot find associate.txt"<<endl;
            exit(0);
        }
        while( !fin.eof())
        {
            _odometry.push_back( readOdometry(fin) );
        }
        _odo_this = _odo_last = _odometry[ _index-1 ];
    }

    //处理首帧
    readimage();
    _lastRGB = _currRGB.clone();
    _currKF.id = 0;
    _currKF.frame_index = _index;
    _currKF.planes.push_back(extractKPandDesp(_currRGB, _currDep));

    _keyframes.push_back( _currKF );
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    VertexSE3* v = new VertexSE3();
    v->setId( _currKF.id );
    if (_use_odometry)
        v->setEstimate( _odo_this );
    else
        v->setEstimate( _robot );
    v->setFixed( true );
    opt.addVertex( v );
    _index ++;
    cout<<"********************"<<endl;
}

int GraphicEnd2::run()
{
    cout<<"********************"<<endl;
    _present.planes.clear();

    readimage();
    _present.planes.push_back( extractKPandDesp( _currRGB, _currDep ) );

    RESULT_OF_MULTIPNP result = multiPnP( _currKF.planes, _present.planes );
    Eigen::Isometry3d T = result.T.inverse();

    if (T.matrix() == Eigen::Isometry3d::Identity().matrix())
    {
        //匹配失败
        cout<<BOLDRED"This frame is lost"<<RESET<<endl;
        _lost++;
    }
    else if (result.norm > _max_pos_change)
    {
        //生成新关键帧
        _robot = T*_kf_pos;
        generateKeyFrame(T);
        if (_loop_closure_detection == true)
            loopClosure();
        _lost = 0;
    }
    else
    {
        //位置变化小
        _robot = T*_kf_pos;
        _lost = 0;
    }

    if (_lost > _lost_frames)
    {
        cerr<<"the robot is lost, perform lost recovery"<<endl;
        lostRecovery();
    }

    cout<<RED"keyframe size = "<<_keyframes.size()<<RESET<<endl;
    _index++;
    
    if (_use_odometry )
    {
        _odo_this = _odometry[_index - 1];
    }
        
    return  0;
}

PLANE GraphicEnd2::extractKPandDesp( Mat& rgb, Mat& dep)
{
    PLANE p;
    p.kp = extractKeypoints( rgb );
    compute3dPosition( p, dep );
    p.desp = extractDescriptor( rgb, p.kp );

    cout<<"Plane kp size = "<<p.kp_pos.size()<<endl;
    return p;
}

void GraphicEnd2::compute3dPosition( PLANE& plane, Mat depth )
{
    vector<KeyPoint> kps;
    for (size_t i=0; i<plane.kp.size(); i++)
    {
        double u = plane.kp[i].pt.x, v = plane.kp[i].pt.y;
        unsigned short d = depth.at<unsigned short>(round(v), round(u));
        if (d == 0)
        {
            //plane.kp_pos.push_back(Point3f( 0,0,0 ) );
            continue;
        }
        kps.push_back( plane.kp[i]);
        double z = double(d)/camera_factor;
        double x = ( u - camera_cx) * z / camera_fx;
        double y = ( v - camera_cy) * z / camera_fy;
        plane.kp_pos.push_back( Point3f( x, y, z) );
    }
    plane.kp = kps;
}

RESULT_OF_MULTIPNP GraphicEnd2::multiPnP(  vector<PLANE>& plane1, vector<PLANE>& plane2, bool loopclosure, int frame_index, int minimum_inliers )
{
    RESULT_OF_MULTIPNP result;
    cout<<"solving multi PnP"<<endl;
    vector<DMatch> matches = match( plane1[0].desp, plane2[0].desp );
    cout<<"good matches: "<<matches.size()<<endl;
    if (matches.size() == 0)
    {
        return result;
    }

    vector<Point3f> obj; 
    vector<Point2f> img;
    for (size_t i=0; i<matches.size(); i++)
    {
        obj.push_back( plane1[0].kp_pos[matches[i].queryIdx] );
        img.push_back( plane2[0].kp[matches[i].trainIdx].pt );
    }
    double camera_matrix[3][3] = { { camera_fx, 0, camera_cx }, { 0, camera_fy ,camera_cy }, { 0, 0, 1 }};
    Mat cameraMatrix(3,3,CV_64F, camera_matrix);

    cout<<"RANSAC obj size = "<<obj.size()<<", img size = "<<img.size()<<endl;
    Mat rvec, tvec; 
    Mat inliers;
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);

    if (inliers.rows < minimum_inliers )
    {
        cout<<"no enough inliers: "<<inliers.rows<<endl;
        return result;
    }

    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( matches[inliers.at<int>(i,0)] );
    cout<<CYAN<<"multiICP::inliers = "<<inliers.rows<<RESET<<endl;
    
    if (loopclosure == false)
    {
        Mat image_matches;
        drawMatches(_lastRGB, plane1[0].kp, _currRGB, plane2[0].kp, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
        imshow("match", image_matches);
        waitKey(_step_time);
    }
    else
    {
        Mat image_matches;
        stringstream ss;
        ss<<_rgbPath<<frame_index<<".png";
        
        Mat rgb = imread( ss.str(), 0);
        drawMatches( rgb, plane1[0].kp, _currRGB, plane2[0].kp, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
        imshow("match", image_matches);
        waitKey(_step_time);
    }
    
    Eigen::Isometry3d T = Isometry3d::Identity();
    double normofTransform = min(norm(rvec), 2*M_PI-norm(rvec)) + norm(tvec);
    cout<<RED<<"norm of Transform = "<<normofTransform<<RESET<<endl;
    result.norm = fabs(normofTransform );
    if (normofTransform > _error_threshold)
    {
        return result;
    }
    Mat R;
    Rodrigues( rvec, R );
    Eigen:Matrix3d r;
    cv2eigen(R, r);
    
    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    T(0,3) = tvec.at<double> (0,0); T(1,3) = tvec.at<double>(0,1); T(2,3) = tvec.at<double>(0,2);
    result.T = T;
    result.inliers = inliers.rows;

    return result;
}
