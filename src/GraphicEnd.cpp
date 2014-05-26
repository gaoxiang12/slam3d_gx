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

//PCL
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

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

GraphicEnd::GraphicEnd() :
    _currCloud( new PointCloud( )),
    _lastCloud( new PointCloud( ))
{
    g_pParaReader = new ParameterReader(parameter_file_addr);
    initModule_nonfree();
    _detector = FeatureDetector::create( g_pParaReader->GetPara("detector_name") );
    _descriptor = DescriptorExtractor::create( g_pParaReader->GetPara("descriptor_name") );

    _robot = Isometry3d::Identity();
}

GraphicEnd::~GraphicEnd()
{
    delete g_pParaReader;
}

void GraphicEnd::init(SLAMEnd* pSLAMEnd)
{
    cout<<"Graphic end init..."<<endl;

    _pSLAMEnd = pSLAMEnd;
    _index = atoi( g_pParaReader->GetPara("start_index").c_str() );
    _rgbPath = g_pParaReader->GetPara("data_source")+string("/rgb_index/");
    _depPath = g_pParaReader->GetPara("data_source")+string("/dep_index/");
    _pclPath = g_pParaReader->GetPara("data_source")+string("/pcd/");
    _loops = 0;
    _success = false;
    _step_time = atoi(g_pParaReader->GetPara("step_time").c_str());
    _distance_threshold = atof( g_pParaReader->GetPara("distance_threshold").c_str() );
    _min_error_plane = atof( g_pParaReader->GetPara("min_error_plane").c_str() );
    _match_min_dist = atof( g_pParaReader->GetPara("match_min_dist").c_str() );
    _percent = atof( g_pParaReader->GetPara("plane_percent").c_str() );
    _max_pos_change = atof( g_pParaReader->GetPara("max_pos_change").c_str());
    
    //读取首帧图像并处理
    readimage();  
    _currKF.id = 0;
    _currKF.planes = extractPlanes( _currCloud ); //抓平面
    for ( size_t i=0; i<_currKF.planes.size(); i++ )
    {
        generateImageOnPlane( _currRGB, _currKF.planes[i],  _currDep );
        _currKF.planes[i].kp = extractKeypoints(_currKF.planes[i].image);
        _currKF.planes[i].desp = extractDescriptor( _currRGB, _currKF.planes[i].kp );
        compute3dPosition( _currKF.planes[i], _currDep );
    }
    _index ++;

    cout<<"********************"<<endl;
}

int GraphicEnd::run()
{
    //清空present并读取新的数据
    _present.planes.clear();
    
    readimage();
    
    //处理present
    _present.planes = extractPlanes( _currCloud );
    for ( size_t i=0; i<_present.planes.size(); i++ )
    {
        generateImageOnPlane( _currRGB, _present.planes[i], _currDep );
        _present.planes[i].kp = extractKeypoints( _present.planes[i].image );
        _present.planes[i].desp = extractDescriptor( _currRGB, _present.planes[i].kp );
        compute3dPosition( _present.planes[i], _currDep);
    }

    // 求解present到current的变换矩阵
    Eigen::Isometry3d T = multiPnP( _currKF.planes, _present.planes );
    // 如果平移和旋转超过一个阈值，则定义新的关键帧
    Eigen::Vector3d rpy = T.rotation().eulerAngles(0, 1, 2);
    Eigen::Vector3d trans = T.translation();
    double norm = rpy.norm() + trans.norm();
    cout<<RED<<"norm of T = "<<norm<<RESET<<endl;
    if (norm > _max_pos_change)
    {
        //生成一个新的关键帧
        generateKeyFrame();
    }

    _index ++;
    return 1;
}

int GraphicEnd::readimage()
{
    cout<<"loading image "<<_index<<endl;
    //读取灰度图,深度图和点云
    ss<<_rgbPath<<_index<<".png";
    _currRGB = imread(ss.str(), 0);
    ss.str("");
    ss.clear();

    //imshow("rgb",_currRGB);
    //waitKey(0);
    
    ss<<_depPath<<_index<<".png";
    _currDep = imread(ss.str(), -1);
    ss.str("");
    ss.clear();
    ss<<_pclPath<<_index<<".pcd";
    pcl::io::loadPCDFile(ss.str(), *_currCloud);
    cout<<"pointcloud size = "<<_currCloud->points.size()<<endl;
    ss.str("");
    ss.clear();
    return 0;
}

void GraphicEnd::generateKeyFrame()
{
    cout<<BOLDGREEN<<"GraphicEnd::generateKeyFrame"<<RESET<<endl;
    _keyframes.push_back( _currKF );

    //当present中的数据存储到current中
    _currKF.id ++;
    _currKF.planes = _present.planes;
    
}

vector<PLANE> GraphicEnd::extractPlanes( PointCloud::Ptr cloud)
{
    cout<<"GraphicEnd::extractPlane..."<<endl;
    vector<PLANE> planes;
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients() );
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices );

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients( true );

    seg.setModelType( pcl::SACMODEL_PLANE );
    seg.setMethodType( pcl::SAC_RANSAC );
    seg.setDistanceThreshold( _distance_threshold );

    int n = cloud->points.size();
    int i=0;

    PointCloud::Ptr tmp (new PointCloud());
    pcl::copyPointCloud(*cloud, *tmp);

    while( tmp->points.size() > _percent*n )
    {
        seg.setInputCloud(tmp);
        seg.segment( *inliers, *coefficients );
        if (inliers->indices.size() == 0)
        {
            break;
        }
        PLANE p;
        p.coff = *coefficients;
        
        if ( coefficients->values[3] < 0)
        {
            for (int i=0; i<4; i++)
                p.coff.values[i] = -p.coff.values[i];
        }
        
        planes.push_back(p);
        cout<<"Coff: "<<p.coff.values[0]<<","<<p.coff.values[1]<<","<<p.coff.values[2]<<","<<p.coff.values[3]<<endl;
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud( tmp );
        extract.setIndices( inliers );
        extract.setNegative( true );
        extract.filter( *tmp );
        i++;
    }

    cout<<"Total planes: "<<i<<endl;
    return planes;
}

void GraphicEnd::generateImageOnPlane( Mat rgb, PLANE& plane, Mat depth)
{
    cout<<"GraphicEnd::generateImageOnPlane"<<endl;
    plane.image = rgb.clone();
    int rows = plane.image.rows, cols = plane.image.cols;
    for (int j=0; j<rows; j++)
    {
        uchar* data = plane.image.ptr<uchar> (j); //行指针
        ushort* depData = depth.ptr<ushort> (j);
        for (int i=0; i<cols; i++)
        {
            ushort d = depData[i];
            if (d == 0)
            {
                data[i] = 0; //置0
                continue;
            }
            double z = double(d)/camera_factor;
            double x = ( i - camera_cx) * z / camera_fx;
            double y = ( j - camera_cy) * z / camera_fy;
            
            double e = plane.coff.values[0]*x + plane.coff.values[1]*y + plane.coff.values[2]*z + plane.coff.values[3];
            e*=e;
            if ( e < _min_error_plane )
            {
            }
            else
                data[i] = 0;
        }
    }

    Mat dst;
    equalizeHist( plane.image, dst );
    plane.image = dst.clone();
    //imshow("image", plane.image);
    //waitKey(0);
}

void GraphicEnd::compute3dPosition( PLANE& plane, Mat depth)
{
    for (size_t i=0; i<plane.kp.size(); i++)
    {
        double u = plane.kp[i].pt.x, v = plane.kp[i].pt.y;
        unsigned short d = depth.at<unsigned short>(round(v), round(u));
        if (d == 0)
        {
            plane.kp_pos.push_back( Point3f(0,0,0) );
            continue;
        }
        double z = double(d)/camera_factor;
        double x = ( u - camera_cx) * z / camera_fx;
        double y = ( v - camera_cy) * z / camera_fy;
        plane.kp_pos.push_back(Point3f( x, y, z) );
    }
}

vector<DMatch> GraphicEnd::match( vector<PLANE>& p1, vector<PLANE>& p2 )
{
    cout<<"GraphicEnd::match two planes"<<endl;
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    cv::Mat des1(p1.size(), 4, CV_32F), des2(p2.size(), 4, CV_32F);
    for (size_t i=0; i<p1.size(); i++)
    {
        pcl::ModelCoefficients c = p1[i].coff;
        float m[1][4] = { c.values[0], c.values[1], c.values[2], c.values[3] };
        Mat mat = Mat(1,4, CV_32F, m);
        mat.row(0).copyTo( des1.row(i) );
    }

    for (size_t i=0; i<p2.size(); i++)
    {
        pcl::ModelCoefficients c = p2[i].coff;
        float m[1][4] = { c.values[0], c.values[1], c.values[2], c.values[3] };
        Mat mat = Mat(1,4, CV_32F, m);
        mat.row(0).copyTo( des2.row(i) );
    }

    matcher.match( des1, des2, matches);
    double max_dist = 0, min_dist = 100;
    for (int i=0; i<des1.rows; i++)
    {
        double dist = matches[ i ].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    //choose good matches
    vector<DMatch> good_matches;
    
    for (int i=0; i<des1.rows; i++)
    {
        if (matches[ i ].distance <= max(2*min_dist, _match_min_dist))
        {
            good_matches.push_back(matches[ i ]);
        }
    }
    return good_matches;
}

vector<DMatch> GraphicEnd::match( Mat desp1, Mat desp2 )
{
    cout<<"GraphicEnd::match two desp"<<endl;
    FlannBasedMatcher matcher;
    vector<DMatch> matches;

    if (desp1.empty() || desp2.empty())
    {
        return matches;
    }
    
    matcher.match( desp1, desp2, matches);
    double max_dist = 0, min_dist = 100;
    for (size_t i=0; i<matches.size(); i++)
    {
        double dist = matches[ i ].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    vector<DMatch> good_matches;
    
    for (size_t i=0; i<matches.size(); i++)
    {
        if (matches[ i ].distance <= max(2*min_dist, match_min_dist))
        {
            good_matches.push_back(matches[ i ]);
        }
    }

    return good_matches;
}

Eigen::Isometry3d GraphicEnd::pnp( PLANE& p1, PLANE& p2)
{
    vector<DMatch> matches = match( p1.desp, p2.desp );
    cout<<"good matches: "<<matches.size()<<endl;

    vector<Point3f> obj; 
    vector<Point2f> img;
    for (size_t i=0; i<matches.size(); i++)
    {
        obj.push_back( p1.kp_pos[matches[i].queryIdx] );
        img.push_back( p2.kp[matches[i].trainIdx].pt );
    }
    double camera_matrix[3][3] = { { camera_fx, 0, camera_cx }, { 0, camera_fy ,camera_cy }, { 0, 0, 1 }};
    Mat cameraMatrix(3,3,CV_64F, camera_matrix);

    Mat rvec, tvec; 

    Mat inliers;
    cout<<"calling solvePnPRansac"<<endl;
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);
    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( matches[inliers.at<int>(i,0)] );
    
    cout<<"inliers = "<<inliers.rows<<endl;
    if (inliers.rows < 4)
    {
        cerr<<"No enough inliers."<<endl;
    }
    Mat image_matches;
    drawMatches(p1.image, p1.kp, p2.image, p2.kp, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    imshow("match", image_matches);
    waitKey(_step_time);

    Eigen::Isometry3d T = Isometry3d::Identity();
    // 旋转向量转换成旋转矩阵
    Mat R;
    Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    cv2eigen(R, r);
    
    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle * trans;
    
    return T;
}

Eigen::Isometry3d GraphicEnd::multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2)
{
    cout<<"solving multi PnP"<<endl;
    vector<DMatch> matches = match( plane1, plane2 );
    cout<<"matches of two planes: "<<matches.size()<<endl;
    vector<Eigen::Isometry3d> transforms;
    for (size_t i=0; i<matches.size(); i++)
    {
        cout<<"planes 1:"<<matches[i].queryIdx<<", planes 2:"<<matches[i].trainIdx<<endl;
        Eigen::Isometry3d t = pnp( plane1[matches[i].queryIdx], plane2[matches[i].trainIdx] );
        cout<<"T"<<i<<" = "<<endl;
        cout<<t.matrix()<<endl;
        transforms.push_back(t);
    }
    //构造本地的图来求解帧间匹配
    
    SparseOptimizer* opt = new SparseOptimizer();
    opt->setVerbose( false );
    opt->setAlgorithm( _pSLAMEnd->solver );
    
    //两个顶点
    VertexSE3* v1 = new VertexSE3();
    VertexSE3* v2 = new VertexSE3();
    v1->setId(0);    v2->setId(1);
    v1->setEstimate( Eigen::Isometry3d::Identity() );
    v2->setEstimate( Eigen::Isometry3d::Identity() );
    v1->setFixed( true );
    opt->addVertex( v1 );
    opt->addVertex( v2 );
    //边
    vector<EdgeSE3*> edges;
    for (size_t i=0; i<transforms.size(); i++)
    {
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt->vertex(0);
        edge->vertices() [1] = opt->vertex(1);
        edge->setMeasurement( transforms[i] );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        //边的信息矩阵暂时不知道怎么定，先取成精度为0.1米，则取逆之后应是100
        information(0, 0) = information(1,1) = information(2,2) = 100; 
        information(3,3) = information(4,4) = information(5,5) = 100; 
        edge->setInformation( information );
        edge->setRobustKernel( _pSLAMEnd->_robustKernel );
        opt->addEdge( edge );
        edges.push_back( edge );
    }
    //求解
    opt->setVerbose( true );
    opt->initializeOptimization();
    opt->optimize( 10 );

    VertexSE3* final = dynamic_cast<VertexSE3*> (opt->vertex(1));
    Eigen::Isometry3d esti = final->estimate();
    cout<<"result of local optimization: "<<endl;
    cout<<esti.matrix()<<endl;

    return esti;
}

////////////////////////////////////////
//SLAMEnd

void SLAMEnd::setupLocalOptimizer()
{
}
