/* ****************************************
 * exp1.cpp -实验一的实现代码
 * xiang, Gao, 2014.9.14
 *****************************************/

#include "../const.h"
#include "../ParameterReader.h"

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
using namespace g2o;
using namespace cv;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//data
Mat rgb_1, rgb_2, dep_1, dep_2;
PointCloud::Ptr cloud_1( new PointCloud), cloud_2(new PointCloud);
struct PLANE
{
    pcl::ModelCoefficients coff;  //a,b,c,d
    vector<KeyPoint> kp;             // keypoints
    vector<Point3f> kp_pos;        // 3d position of keypoints
    Mat desp;                               // descriptor
    Mat image;                             // grayscale image with mask
};

Eigen::Isometry3d Matching( char* featureName, char* despName);
Eigen::Isometry3d MatchingPlanar( char* featureName, char* despName );

//子函数
//提取平面 
vector<PLANE> extractplanes( PointCloud::Ptr cloud, Mat& rgb, Mat& dep );
//平面化特征的PnP
Eigen::Isometry3d planarPnP( vector<PLANE>& , vector<PLANE>& );

//匹配两组平面，以法向量为特征
vector<DMatch> match( vector<PLANE>& p1, vector<PLANE>& p2);
//基本PnP
vector<DMatch> pnp( PLANE& p1, PLANE& p2 ); 
//提取关键点
inline vector<KeyPoint> extractKeypoints(Mat& image, char* featureName)        
{
    vector<KeyPoint> kp;
    Ptr<FeatureDetector> detector = FeatureDetector::create(featureName);
    detector->detect(image, kp);
    return kp;
}
//提取描述子
inline Mat extractDescriptor( Mat& image, vector<KeyPoint>& kp, char* despName) 
{
    Mat desp;
    Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(despName);
    descriptor->compute( image, kp, desp);
    return desp;
}
//匹配描述子
vector<DMatch> match( Mat desp1, Mat desp2 );
//计算3d位置 
void compute3dPosition( PLANE& plane, Mat& depth)
{
    for (size_t i=0; i<plane.kp.size(); i++)
    {
        double u = plane.kp[i].pt.x, v = plane.kp[i].pt.y;
        unsigned short d = depth.at<unsigned short>(round(v), round(u));
        //if (d == 0)
        {
            double
            a = plane.coff.values[0],
            b = plane.coff.values[1],
            c = plane.coff.values[2],
            e = plane.coff.values[3];
            double k1 = (u - camera_cx) / camera_fx, k2 = ( v - camera_cy ) / camera_fy;
            double z = -e / (a*k1 + b*k2 + c);
            double x = k1*z, y = k2*z;
            plane.kp_pos.push_back(Point3f( x, y, z) );
            continue;
        }
        double z = double(d)/camera_factor;
        double x = ( u - camera_cx) * z / camera_fx;
        double y = ( v - camera_cy) * z / camera_fy;
        plane.kp_pos.push_back( Point3f( x, y, z) );
    }
}

void usage()
{
    cout<<"usage: exp1 detector descriptor"<<endl;
    return;
}

int main( int argc, char** argv )
{
    if (argc != 3)
    {
        usage();
        return -1;
    }

    //准备工作 
    cout<<"Experiment 1: test planar feature matching."<<endl;
    cout<<"Loading data..."<<endl;
    
    g_pParaReader = new ParameterReader( parameter_file_addr );
    initModule_nonfree();
    
    rgb_1 = imread( "data/exp1/rgb_1.png", 0);
    rgb_2 = imread( "data/exp1/rgb_2.png", 0);
    dep_1 = imread( "data/exp1/dep_1.png", -1);
    dep_2 = imread( "data/exp1/dep_2.png", -1);
    pcl::io::loadPCDFile( "data/exp1/pcd_1.pcd", *cloud_1 );
    pcl::io::loadPCDFile( "data/exp1/pcd_2.pcd", *cloud_2 );

    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName( "z" );
    pass.setFilterLimits(0.0, 6.0);
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize( 0.02, 0.02, 0.02 );
    
    PointCloud::Ptr tmp ( new PointCloud() );
    pass.setInputCloud( cloud_1 );
    pass.filter(*tmp);
    voxel.setInputCloud( tmp );
    voxel.filter( *cloud_1 );

    pass.setInputCloud( cloud_2 );
    pass.filter(*tmp);
    voxel.setInputCloud( tmp );
    voxel.filter( *cloud_2 );

    //读取groundtruth
    ifstream fin("./data/exp1/groundtruth.txt");
    g2o::VertexSE3 v;
    double data[7] ;
    for (int i=0; i<7; i++)
        fin>>data[i];
    v.setEstimateData( data );
    Eigen::Isometry3d T1 = v.estimate();
    for (int i=0; i<7; i++)
        fin>>data[i];
    v.setEstimateData( data );
    Eigen::Isometry3d T2 = v.estimate();
    
    /* 测试欧拉角的转换是否正确
    Eigen::Isometry3d Ttest;
    Eigen::AngleAxisd r0_t(rpy[0], Eigen::Vector3d::UnitZ()),
        r1_t( rpy[1], Eigen::Vector3d::UnitX()),
        r2_t( rpy[2], Eigen::Vector3d::UnitZ());
    Ttest = r0_t*r1_t*r2_t;

    cout<<"True:"<<endl<<v.estimate().matrix()<<endl;
    cout<<"Test:"<<endl<<Ttest.matrix()<<endl;
    */
    Eigen::Isometry3d Tr = T2*T1.inverse();
    cout<<"Tr = "<<endl<<Tr.matrix()<<endl;
    Eigen::Vector3d rpy = Tr.rotation().eulerAngles(2, 0, 2);
    Eigen::Isometry3d T_truth;
    
    Eigen::AngleAxisd r0(rpy[0], -Eigen::Vector3d::UnitY()),
        r1( rpy[1], Eigen::Vector3d::UnitZ()),
        r2( rpy[2], -Eigen::Vector3d::UnitY());
    T_truth = r0*r1*r2;
    T_truth.matrix()(0,3) = -Tr.matrix()(1,3);
    T_truth.matrix()(1,3) = -Tr.matrix()(2,3);
    T_truth.matrix()(2,3) = Tr.matrix()(0,3);
    
    /*
    rpy = v.estimate().rotation().eulerAngles(2, 0, 2);
    r0 = Eigen::AngleAxisd(rpy[2], -Eigen::Vector3d::UnitY());
    r1 = Eigen::AngleAxisd( rpy[1], Eigen::Vector3d::UnitZ());
    r2 = Eigen::AngleAxisd( rpy[2], -Eigen::Vector3d::UnitY());
    T2 = r0*r1*r2;
    T2.matrix()(0,3) = -data[1];
    T2.matrix()(1,3) = -data[2];
    T2.matrix()(2,3) = data[0];
    */
    fin.close();

    cout<<"T1 = "<<endl<<T1.matrix()<<endl;
    cout<<"T2 = "<<endl<<T2.matrix()<<endl;
    cout<<"groundtruth = "<<endl;
    cout<<T_truth.matrix()<<endl;
    cout<<"Load ok."<<endl;

    //用指定特征进行匹配
    Eigen::Isometry3d T = Matching( argv[1], argv[2] );
    Eigen::Isometry3d Tp = MatchingPlanar( argv[1], argv[2] );

    //分析误差
    delete g_pParaReader;

    cout<<"ordinary match: "<<endl<<(T).matrix()<<endl;
    cout<<"planar match: "<<endl<<(Tp).matrix()<<endl;
    
    Eigen::Isometry3d Terror = T_truth*T.inverse();
    cout<<"error matrix: "<<endl<<Terror.matrix()<<endl;
    rpy = Terror.rotation().eulerAngles( 2, 0, 2);
    Eigen::Vector3d trans( Terror.matrix()(0,3), Terror.matrix()(1,3), Terror.matrix()(2,3));

    cout<<"norm of error: "<<trans.norm()<<", "<<rpy.norm()<<endl;
    return 0;
}

Eigen::Isometry3d Matching( char* featureName, char* despName)
{
    cout<<"Matching using "<<featureName<<endl;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    Ptr<FeatureDetector> detector = FeatureDetector::create(featureName);
    if (detector.empty())
    {
        cerr<<"fail to create detector"<<endl;
        return T;
    }

    vector<KeyPoint> kp1, kp2;
    detector->detect( rgb_1, kp1);
    detector->detect( rgb_2, kp2);

    cout<<"kp1 size="<<kp1.size()<<", kp2 size = "<<kp2.size()<<endl;

    Mat des1, des2;
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create(despName);

    descriptor_extractor->compute( rgb_1, kp1, des1 );
    descriptor_extractor->compute( rgb_2, kp2, des2 );

    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    
    matcher.match( des1, des2, matches);

    
    double max_dist = 0, min_dist = 1000;
    for (int i=0; i<des1.rows; i++)
    {
        double dist = matches[ i ].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    vector<DMatch> good_matches;
    double match_min_dist = 5.0;
    for (int i=0; i<des1.rows; i++)
    {
        if (matches[ i ].distance <= max(2*min_dist, match_min_dist))
        {
            good_matches.push_back(matches[ i ]);
        }
        }
    
    //vector<DMatch> good_matches = matches;
    cout<<"good matches: "<<good_matches.size()<<endl;
    
    vector<Point3f> obj;
    vector<Point2f> img;
    //选取带有深度信息的特征点
    vector<KeyPoint> kp1_f, kp2_f;
    
    for (size_t i=0; i<good_matches.size(); i++)
    {
        KeyPoint kp = kp1[good_matches[i].queryIdx];
        double u = kp.pt.x, v = kp.pt.y;
        unsigned short d = dep_1.at<unsigned short>(round(v), round(u));
        if (d == 0)
        {
            continue;
        }
        double z = double(d)/camera_factor;
        double x = ( u - camera_cx) * z / camera_fx;
        double y = ( v - camera_cy) * z / camera_fy;
        obj.push_back( Point3f( x, y, z) );
        img.push_back( kp2[good_matches[i].trainIdx].pt );
        kp1_f.push_back( kp );
        kp2_f.push_back( kp2[good_matches[i].trainIdx]);
    }

    cout<<"camera factor = "<<camera_factor<<endl;
    
    double camera_matrix[3][3] = { { camera_fx, 0, camera_cx }, { 0, camera_fy ,camera_cy }, { 0, 0, 1 }};
    Mat cameraMatrix(3,3,CV_64F, camera_matrix);
    Mat rvec, tvec; 
    Mat inliers;
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);

    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( good_matches[inliers.at<int>(i,0)] );
   

    cout<<"inlier: "<<inliers.rows<<endl;
    //Eigen::Isometry3d T = Isometry3d::Identity();

    // 旋转向量转换成旋转矩阵
    Mat R;
    Rodrigues( rvec, R );
    Eigen:Matrix3d r;
    cv2eigen(R, r);
    
    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    T(0,3) = tvec.at<double> (0,0); T(1,3) = tvec.at<double>(0,1); T(2,3) = tvec.at<double>(0,2);

    
    //提取inliers作图
    Mat image_matches;
    drawMatches( rgb_1, kp1, rgb_2, kp2, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    imshow("match", image_matches);
    waitKey(0);
    return T;
}

Eigen::Isometry3d MatchingPlanar( char* featureName, char* despName )
{
    //Step 1 提取平面
    vector<PLANE> p1 = extractplanes( cloud_1, rgb_1, dep_1 );
    vector<PLANE> p2 = extractplanes( cloud_2, rgb_2, dep_2 );

    for (size_t i=0; i<p1.size(); i++)
    {
        p1[i].kp = extractKeypoints( p1[i].image, featureName );
        p1[i].desp = extractDescriptor( rgb_1, p1[i].kp, despName );
        compute3dPosition( p1[i], dep_1);
    }

    for (size_t i=0; i<p2.size(); i++)
    {
        p2[i].kp = extractKeypoints( p2[i].image, featureName );
        p2[i].desp = extractDescriptor( rgb_2, p2[i].kp, despName );
        compute3dPosition( p2[i], dep_2);
    }

    vector<DMatch> matches = match( p1, p2 );
    cout<<"matches of two planes: "<<matches.size()<<endl;

    vector<Point3f> obj; 
    vector<Point2f> img;

    vector<KeyPoint> kp1, kp2;
    vector<DMatch> match_show;
    int n=0;
    for (size_t i=0; i<matches.size(); i++)
    {
        vector<DMatch> kpMatches = pnp( p1[matches[i].queryIdx], p2[matches[i].trainIdx] );
        for (size_t j=0; j<kpMatches.size(); j++)
        {
            obj.push_back( p1[matches[i].queryIdx].kp_pos[kpMatches[j].queryIdx] );
            img.push_back( p2[matches[i].trainIdx].kp[kpMatches[j].trainIdx].pt );
            kp1.push_back( p1[matches[i].queryIdx].kp[kpMatches[j].queryIdx]);
            kp2.push_back( p2[matches[i].trainIdx].kp[kpMatches[j].trainIdx]);

            match_show.push_back( DMatch(n, n, kpMatches[j].distance) );
            n++;
        }
    }

    if (obj.empty())
    {
        cout<<"object is empty"<<endl;
        return Eigen::Isometry3d::Identity();
    }
    
    double camera_matrix[3][3] = { { camera_fx, 0, camera_cx }, { 0, camera_fy ,camera_cy }, { 0, 0, 1 }};
    Mat cameraMatrix(3,3,CV_64F, camera_matrix);
    Mat rvec, tvec; 
    Mat inliers;
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);

    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( match_show[inliers.at<int>(i,0)] );
    
    cout<<"multiICP::inliers = "<<inliers.rows<<endl;
    cout<<"matches: "<<match_show.size()<<endl;
    Mat image_matches;
    drawMatches( rgb_1, kp1, rgb_2, kp2, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    imshow("match planar", image_matches);
    waitKey( 0 );

    cout<<"rvec = "<<rvec<<endl<<"tvec = "<<tvec<<endl;
    Eigen::Isometry3d T = Isometry3d::Identity();
    // 旋转向量转换成旋转矩阵
    Mat R;
    Rodrigues( rvec, R );
    Eigen:Matrix3d r;
    cv2eigen(R, r);
    
    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    T(0,3) = tvec.at<double> (0,0); T(1,3) = tvec.at<double>(0,1); T(2,3) = tvec.at<double>(0,2);
    
    return T;
}

vector<PLANE> extractplanes( PointCloud::Ptr cloud, Mat& rgb, Mat& dep )
{
    cout<<"extracting planes"<<endl;
    vector<PLANE> planes;
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients() );
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices ); 

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients( true );
    seg.setModelType( pcl::SACMODEL_PLANE );
    seg.setMethodType( pcl::SAC_RANSAC );
    seg.setDistanceThreshold( atof(g_pParaReader->GetPara("distance_threshold").c_str() ));

    int n = cloud->points.size();
    int i=0;

    PointCloud::Ptr tmp (new PointCloud()); //储存剩下的数据
    pcl::copyPointCloud(*cloud, *tmp);

    double _percent = atof( g_pParaReader->GetPara("plane_percent").c_str());
    double _max_planes = atof( g_pParaReader->GetPara("max_planes").c_str());
    while( tmp->points.size() > _percent*n )
    {
        seg.setInputCloud(tmp);
        seg.segment( *inliers, *coefficients );
        if (inliers->indices.size() == 0) //没有剩余的平面
        {
            break;
        }
        PLANE p;
        p.coff = *coefficients;
        
        if ( coefficients->values[3] < 0)  //归一化
        {
            for (int i=0; i<4; i++)
                p.coff.values[i] = -p.coff.values[i];
        }
        
        cout<<"Coff: "<<p.coff.values[0]<<","<<p.coff.values[1]<<","<<p.coff.values[2]<<","<<p.coff.values[3]<<endl;
        pcl::ExtractIndices<PointT> extract;
        PointCloud::Ptr plane_cloud( new PointCloud());

        extract.setInputCloud( tmp );
        extract.setIndices( inliers );
        extract.setNegative( false );
        extract.filter( *plane_cloud ); //把选中的点滤出
        p.image = Mat( 480, 640, CV_8UC1, Scalar::all(0));

        cout<<"plane "<<i<<" inlier size = "<<plane_cloud->points.size()<<endl;
        int block = 4;
        for (size_t j=0; j<plane_cloud->points.size(); j++)
        {
            //生成该平面对应的图像
            PointT pt = plane_cloud->points[j];
            int u = round( pt.x*camera_fx/pt.z + camera_cx );
            int v = round( pt.y*camera_fy/pt.z + camera_cy );
            for (int k=-block; k<block+1; k++)
                for (int l=-block; l<block+1; l++)
                {
                    if (v+k < 0 || v+k>=480 || u+l<0 || u+l >=640)
                        continue;
                    p.image.ptr(v+k)[u+l] = rgb.ptr(v+k)[u+l];
                }
            //p.image.ptr(v) [u] = rgb.ptr(v)[u];
        }

        equalizeHist( p.image, p.image);
        //imshow( "image from pointcloud", p.image);
        //waitKey(0);
        
        extract.setNegative( true );
        extract.filter( *tmp ); //把没有被选中的点滤出
        i++;
        planes.push_back(p);

        if (i == _max_planes)
            break;
    }

    cout<<"Total planes: "<<i<<endl;
    return planes;
}

Eigen::Isometry3d planarPnP( vector<PLANE>& plane1, vector<PLANE>& plane2 )
{
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    vector<DMatch> matches = match( plane1, plane2 );
    vector<Point3f> obj; 
    vector<Point2f> img;
    vector<KeyPoint> kp1, kp2;
    vector<DMatch> match_show;
    int n=0;
    for (size_t i=0; i<matches.size(); i++)
    {
        vector<DMatch> kpMatches = pnp( plane1[matches[i].queryIdx], plane2[matches[i].trainIdx] );
        for (size_t j=0; j<kpMatches.size(); j++)
        {
            obj.push_back( plane1[matches[i].queryIdx].kp_pos[kpMatches[j].queryIdx] );
            img.push_back( plane2[matches[i].trainIdx].kp[kpMatches[j].trainIdx].pt );
            kp1.push_back( plane1[matches[i].queryIdx].kp[kpMatches[j].queryIdx]);
            kp2.push_back( plane2[matches[i].trainIdx].kp[kpMatches[j].trainIdx]);
            match_show.push_back( DMatch(n, n, kpMatches[j].distance) );
            n++;
        }
    }

    if (obj.empty())
    {
        cout<<"object is empty"<<endl;
        return result;
    }
    
    double camera_matrix[3][3] = { { camera_fx, 0, camera_cx }, { 0, camera_fy ,camera_cy }, { 0, 0, 1 }};
    Mat cameraMatrix(3,3,CV_64F, camera_matrix);
    Mat rvec, tvec; 
    Mat inliers;
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);

    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( match_show[inliers.at<int>(i,0)] );
    
    cout<<"multiICP::inliers = "<<inliers.rows<<endl;
    cout<<"matches: "<<match_show.size()<<endl;
    Mat image_matches;
    drawMatches( rgb_1, kp1,  rgb_2, kp2, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    imshow("match_planar", image_matches);
    waitKey( 0 );
    
    Eigen::Isometry3d T = Isometry3d::Identity();

    // 旋转向量转换成旋转矩阵
    Mat R;
    Rodrigues( rvec, R );
    Eigen:Matrix3d r;
    cv2eigen(R, r);
    
    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
    T = angle;
    T(0,3) = tvec.at<double> (0,0); T(1,3) = tvec.at<double>(0,1); T(2,3) = tvec.at<double>(0,2);
    return T;
}

vector<DMatch> pnp( PLANE& p1, PLANE& p2)
{
    vector<DMatch> matches = match( p1.desp, p2.desp );
    if (matches.size() == 0)
    {
        return vector<DMatch> ();
    }

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
    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( matches[inliers.at<int>(i,0)] );

    return inlierMatches;
}

vector<DMatch> match( vector<PLANE>& p1, vector<PLANE>& p2 )
{
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

    vector<DMatch> good_matches;
    for (size_t i=0; i<matches.size(); i++)
    {
        if (matches[ i ].distance <= 3*min_dist)
        {
            good_matches.push_back(matches[ i ]);
        }
    }
    return good_matches;
}

vector<DMatch> match( Mat desp1, Mat desp2 )
{
    cout<<"GraphicEnd::match two desp"<<endl;
    FlannBasedMatcher matcher;
    vector<DMatch> matches;

    if (desp1.empty() || desp2.empty())
    {
        return matches;
    }
    double max_dist = 0, min_dist = 100;
    matcher.match( desp1, desp2, matches);

    for (int i=0; i<desp1.rows; i++)
    {
        double dist = matches[ i ].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }


    double _match_min_dist = atof( g_pParaReader->GetPara("match_min_dist").c_str());
    vector<DMatch> good_matches;
    for (size_t i=0; i<matches.size(); i++)
    {
        if (matches[ i ].distance <= max(4*min_dist, _match_min_dist))
        {
            good_matches.push_back(matches[ i ]);
        }
    }
    return good_matches;
}
