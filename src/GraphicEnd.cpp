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

using namespace std;
using namespace cv;

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

void GraphicEnd::init()
{
    cout<<"Graphic end init..."<<endl;
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
    readimage();  //读取首帧图像
    _currKF.id = 0;
    _currKF.planes = extractPlanes( _currCloud ); //抓平面
    for ( size_t i=0; i<_currKF.planes.size(); i++ )
    {
        generateImageOnPlane( _currRGB, _currKF.planes[i],  _currDep );
        _currKF.planes[i].kp = extractKeypoints(_currKF.planes[i].image);
        _currKF.planes[i].desp = extractDescriptor( _currRGB, _currKF.planes[i].kp );
    }
    _index ++;
}

int GraphicEnd::run()
{
    readimage();
    
    return 1;
}

int GraphicEnd::readimage()
{
    cout<<"loading image "<<_index<<endl;
    //读取灰度图,深度图和点云
    stringstream ss;
    ss<<_rgbPath<<_index<<".png";
    _currRGB = imread(ss.str(), 0);
    ss.clear();
    ss<<_depPath<<_index<<".png";
    _currDep = imread(ss.str(), -1);
    ss.clear();
    ss<<_pclPath<<_index<<".pcd";
    pcl::io::loadPCDFile(ss.str(), *_currCloud);
    
    return 0;
}

void GraphicEnd::generateKeyFrame()
{
    cout<<"GraphicEnd::generateKeyFrame"<<endl;
    _keyframes.push_back( _currKF );
    //清空当前关键帧的数据
    _currKF.id ++;
    _currKF.planes.clear();
    
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
            break;
        PLANE p;
        p.coff = *coefficients;
        
        if ( coefficients->values[3] < 0)
        {
            for (int i=0; i<4; i++)
                p.coff.values[i] = -p.coff.values[i];
        }
        
        planes.push_back(p);

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
        uchar* data = rgb.ptr<uchar> (j); //行指针
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
    cout<<"start matching two planes"<<endl;
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
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);
    //vector<DMatch> inlierMatches.clear();
    //for (int i=0; i<inliers.rows; i++)
    //inlierMatches.push_back( matches[inliers.at<int>(i,0)] );
    
    cout<<"inliers = "<<inliers.rows<<endl;
    if (inliers.rows < 4)
    {
        cerr<<"No enough inliers."<<endl;
    }
    //Mat image_matches;
    //drawMatches(rgb1, p1.kp, rgb2, p2.kp, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    //imshow("match", image_matches);
    //waitKey(0);

    Eigen::Isometry3d T = Isometry3d::Identity();
    // 旋转向量转换成旋转矩阵
    Mat R;
    Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    cv2eigen(R, r);

    AngleAxisd angle(r);
    Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = trans* angle;
    
    return T;

}

Eigen::Isometry3d GraphicEnd::multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2)
{
    vector<DMatch> matches = match( plane1, plane2 );
    for (size_t i=0; i<matches.size(); i++)
    {
        cout;
    }
}
