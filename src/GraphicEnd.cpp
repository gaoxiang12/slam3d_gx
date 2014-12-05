/* ****************************************
 * GraphicEnd.cpp
 * 视觉slam的核心算法
 * TODO:
 * 2014.6.24 生成平面图时是否应该做一些形态学运算，补掉里面的洞？
 * 还须一个不用平面特征的对比程序以比较结果
 *****************************************/
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

GraphicEnd::GraphicEnd() :
    _currCloud( new PointCloud( )),
    _lastCloud( new PointCloud( ))
{
    g_pParaReader = new ParameterReader(parameter_file_addr);
    initModule_nonfree();
    _detector = FeatureDetector::create( g_pParaReader->GetPara("detector_name") );
    _descriptor = DescriptorExtractor::create( g_pParaReader->GetPara("descriptor_name") );

    _robot = Isometry3d::Identity();

    srand( (unsigned int) time(0) );
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
    _error_threshold = atof( g_pParaReader->GetPara("error_threshold").c_str() );
    _min_error_plane = atof( g_pParaReader->GetPara("min_error_plane").c_str() );
    _match_min_dist = atof( g_pParaReader->GetPara("match_min_dist").c_str() );
    _percent = atof( g_pParaReader->GetPara("plane_percent").c_str() );
    _max_pos_change = atof( g_pParaReader->GetPara("max_pos_change").c_str());
    _max_planes = atoi( g_pParaReader->GetPara("max_planes").c_str() );
    _loopclosure_frames = atoi( g_pParaReader->GetPara("loopclosure_frames").c_str() );
    _loop_closure_detection = (g_pParaReader->GetPara("loop_closure_detection") == string("yes"))?true:false;
    _loop_closure_error = atof(g_pParaReader->GetPara("loop_closure_error").c_str());
    _loop_closure_inliers = atoi( g_pParaReader->GetPara("loop_closure_inliers").c_str());
    _lost_frames = atoi( g_pParaReader->GetPara("lost_frames").c_str() );
    _robot = _kf_pos = Eigen::Isometry3d::Identity();
    _use_odometry = g_pParaReader->GetPara("use_odometry") == string("yes");
    _error_odometry = atof( g_pParaReader->GetPara("error_odometry").c_str() );
    _z_filter = atof( g_pParaReader->GetPara("z_filter").c_str());
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
    
    //读取首帧图像并处理
    readimage();
    _lastRGB = _currRGB.clone();
    _currKF.id = 0;
    _currKF.frame_index = _index;

    _currKF.planes = extractPlanesAndGenerateImage( _currCloud, _currRGB, _currDep );
    for ( size_t i=0; i<_currKF.planes.size(); i++ )
    {
        _currKF.planes[i].kp = extractKeypoints(_currKF.planes[i].image, _currKF.planes[i].mask);
        _currKF.planes[i].desp = extractDescriptor( _currRGB, _currKF.planes[i].kp );
        compute3dPosition( _currKF.planes[i], _currDep );
    }
    _keyframes.push_back( _currKF );
    //将current存储至全局优化器
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

int GraphicEnd::run()
{
    //清空present并读取新的数据
    static ofstream errorfile("./data/error_of_transform.log");
    cout<<"********************"<<endl;
    _present.planes.clear();
    readimage();
    //处理present
    _present.planes = extractPlanesAndGenerateImage( _currCloud, _currRGB, _currDep );
    
    for ( size_t i=0; i<_present.planes.size(); i++ )
    {
        _present.planes[i].kp = extractKeypoints( _present.planes[i].image, _present.planes[i].mask );
        _present.planes[i].desp = extractDescriptor( _currRGB, _present.planes[i].kp );
        compute3dPosition( _present.planes[i], _currDep);
    }

    // 求解present到current的变换矩阵
    RESULT_OF_MULTIPNP result = multiPnP( _currKF.planes, _present.planes );
    Eigen::Isometry3d T = result.T;
    T = T.inverse();  //好像是反着的
    
    // 如果平移和旋转超过一个阈值，则定义新的关键帧
    if ( T.matrix() == Eigen::Isometry3d::Identity().matrix() )
    {
        //未匹配上
        errorfile<<"9999"<<endl;
        if (_use_odometry)
        {
            _lost++;
        }
        else
        {
            cout<<BOLDRED"This frame lost"<<RESET<<endl;
            //尝试匹配_present 和 _last
            cout<<"Matching last and present."<<endl;
            cout<<"last: "<<_last.frame_index<<endl;
            RESULT_OF_MULTIPNP r = multiPnP( _last.planes, _present.planes );
            if ( r.inliers<_loop_closure_inliers || r.norm > _loop_closure_error )
                _lost++; //该帧失败,丢弃之
            else
            {
                cout<<BOLDGREEN<<"add last as a new keyframe."<<endl;
                //虽然该帧与上一个关键帧匹配不上，但与上一个普通帧对上了。应该给它一个机会。
                _lost = 0;
                RESULT_OF_MULTIPNP rr = multiPnP( _currKF.planes, _last.planes );
                //将上一个普通帧作为关键帧,存入图中
                _currKF.id ++;
                _currKF.planes = _last.planes;
                _currKF.frame_index = _index-1;
                _keyframes.push_back(_currKF );
            
                //将current关键帧存储至全局优化器
                SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
                //顶点
                VertexSE3* v = new VertexSE3();
                v->setId( _currKF.id );
                if (_use_odometry)
                    v->setEstimate( _odo_this );
                else
                    v->setEstimate( Eigen::Isometry3d::Identity() );
                opt.addVertex( v );
                //边
                EdgeSE3* edge = new EdgeSE3();
                edge->vertices()[0] = opt.vertex( _currKF.id - 1 );
                edge->vertices()[1] = opt.vertex( _currKF.id );
                Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
                information(0, 0) = information(2,2) = 100;
                information(1, 1) = 100;
                information(3,3) = information(4,4) = information(5,5) = 100; 
                edge->setInformation( information );
                edge->setMeasurement( rr.T.inverse() );
                opt.addEdge( edge );

                //然后，将present也作为新的关键帧
                generateKeyFrame( r.T.inverse() );
                _last = _present;
            }
        }
    }
    else if (result.norm > _max_pos_change)
    {
        errorfile<<result.norm<<endl;
        //生成一个新的关键帧
        _robot = T * _kf_pos;
        generateKeyFrame(T);
        if (_loop_closure_detection == true)
            loopClosure();
        _lost = 0;
        _last = _present;
    }
    else
    {
        errorfile<<result.norm<<endl;
        //小变化，更新robot位置
        _robot = T * _kf_pos;
        _lost = 0;
        _last = _present;
    }
    if (_lost > _lost_frames)
    {
        cerr<<"the robot lost. Perform lost recovery."<<endl;
        lostRecovery();
        _last = _present;
    }

    _index ++;
    if (_use_odometry)
    {
        _odo_this = _odometry[ _index-1 ];
    }

    _present.frame_index = _index;
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

    ss<<_depPath<<_index<<".png";
    _currDep = imread(ss.str(), -1);
    ss.str("");
    ss.clear();
    
    ss<<_pclPath<<_index<<".pcd";
    pcl::io::loadPCDFile(ss.str(), *_currCloud);
    
    static pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, _z_filter);

    static pcl::VoxelGrid<PointT> voxel;
    double grid = atof(g_pParaReader->GetPara("grid_leaf").c_str());
    voxel.setLeafSize( grid, grid, grid );
    
    PointCloud::Ptr tmp( new PointCloud() );
    pass.setInputCloud(_currCloud);
    pass.filter( *tmp );
    voxel.setInputCloud( tmp );
    voxel.filter(*_currCloud);

    ss.str("");
    ss.clear();
    
    cout<<"load ok."<<endl;
    return 0;
}

void GraphicEnd::generateKeyFrame( Eigen::Isometry3d T )
{
    cout<<BOLDGREEN<<"GraphicEnd::generateKeyFrame"<<RESET<<endl;
    //把present中的数据存储到current中
    _currKF.id ++;
    _currKF.planes = _present.planes;
    _currKF.frame_index = _index;
    _lastRGB = _currRGB.clone();
    _kf_pos = _robot;

    _keyframes.push_back( _currKF );
    
    //将current关键帧存储至全局优化器
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    //顶点
    VertexSE3* v = new VertexSE3();
    v->setId( _currKF.id );
    if (_use_odometry)
        v->setEstimate( _odo_this );
    else
        v->setEstimate( Eigen::Isometry3d::Identity() );
    opt.addVertex( v );
    //边
    EdgeSE3* edge = new EdgeSE3();
    edge->vertices()[0] = opt.vertex( _currKF.id - 1 );
    edge->vertices()[1] = opt.vertex( _currKF.id );
    Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
    information(0, 0) = information(2,2) = 100;
    information(1, 1) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100; 
    edge->setInformation( information );
    edge->setMeasurement( T );
    opt.addEdge( edge );

    if (_use_odometry)
    {
        Eigen::Isometry3d To = _odo_last.inverse()*_odo_this;
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices()[0] = opt.vertex( _currKF.id - 1 );
        edge->vertices()[1] = opt.vertex( _currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(1,1) = information(2,2) = information(3,3) = information(4,4) = information(5,5) = 1/(_error_odometry*_error_odometry);
        edge->setInformation( information );
        edge->setMeasurement( To );
        opt.addEdge( edge );
        _odo_last = _odo_this;
    }
}

vector<PLANE> GraphicEnd::extractPlanesAndGenerateImage( PointCloud::Ptr cloud, Mat& rgb, Mat& dep)
{
    cout<<"extracting planes"<<endl;
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

    PointCloud::Ptr tmp (new PointCloud()); //储存剩下的数据
    pcl::copyPointCloud(*cloud, *tmp);

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
        p.mask = Mat( 480, 640, CV_8UC1, Scalar::all(0));
        int block = 4;
        for (size_t j=0; j<plane_cloud->points.size(); j++)
        {
            //生成该平面对应的图像
            PointT pt = plane_cloud->points[j];
            block = int(-1.0*(pt.z)+10.0);
            block = block>0?block:0;
            int u = round( pt.x*camera_fx/pt.z + camera_cx );
            int v = round( pt.y*camera_fy/pt.z + camera_cy );
            for (int k=-block; k<block+1; k++)
                for (int l=-block; l<block+1; l++)
                {
                    if (v+k < 0 || v+k>=480 || u+l<0 || u+l >=640)
                        continue;
                    p.image.ptr(v+k)[u+l] = rgb.ptr(v+k)[u+l];
                    p.mask.ptr(v+k)[u+l] = 1;
                }
        }

        equalizeHist( p.image, p.image);
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


void GraphicEnd::compute3dPosition( PLANE& plane, Mat depth)
{
    for (size_t i=0; i<plane.kp.size(); i++)
    {
        double u = plane.kp[i].pt.x, v = plane.kp[i].pt.y;
        unsigned short d = depth.at<unsigned short>(round(v), round(u));
        if (d == 0)
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

    return matches;
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

    //return matches;

    vector<DMatch> good_matches;
    for (size_t i=0; i<matches.size(); i++)
    {
        if (matches[ i ].distance <= 3*min_dist)
        {
            good_matches.push_back(matches[ i ]);
        }
    }
    cout<<"good matches: "<<good_matches.size()<<endl;
    return good_matches;
}

vector<DMatch> GraphicEnd::pnp( PLANE& p1, PLANE& p2)
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

    /*
    Mat image_matches;
    drawMatches( p1.image, p1.kp, p2.image, p2.kp, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    imshow("match of plane", image_matches);
    waitKey(_step_time);
    */
    return inlierMatches;
}

//通过若干个平面求解PnP问题，当匹配不上时会返回I
RESULT_OF_MULTIPNP GraphicEnd::multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2, bool loopclosure, int frame_index, int minimum_inliers)
{
    RESULT_OF_MULTIPNP result;
    cout<<"solving multi PnP"<<endl;
    vector<DMatch> matches = match( plane1, plane2 );
    cout<<"matches of two planes: "<<matches.size()<<endl;

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

    cout<<"Multipnp inliers = "<<inliers.rows<<endl;
    result.inliers = inliers.rows;
    if (inliers.rows < minimum_inliers )
        return result;

    cout<<BOLDRED<<"ransac again!"<<RESET<<endl;
    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( match_show[inliers.at<int>(i,0)] );

    vector<Point3f> obj_new;
    vector<Point2f> img_new;
    for (size_t i=0; i<inlierMatches.size(); i++)
    {
        obj_new.push_back( obj[inlierMatches[i].queryIdx]);
        img_new.push_back( img[inlierMatches[i].trainIdx] );
    }
    Mat inliers_new;
    solvePnPRansac(obj_new, img_new, cameraMatrix, Mat(), rvec, tvec, true, 100, 3.0, 100, inliers_new);
    
    Eigen::Isometry3d T = Isometry3d::Identity();
    double normofTransform = fabs(1.0*min(norm(rvec), 2*M_PI-norm(rvec)))+ 0.9*fabs(norm(tvec));
    cout<<RED<<"norm of Transform = "<<normofTransform<<RESET<<endl;
    result.norm = normofTransform;
    if ( result.norm > _error_threshold)
    {
        return result;
    }
    
    if (loopclosure == false)
    {
        Mat image_matches;
        drawMatches(_lastRGB, kp1, _currRGB, kp2, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
        imshow("match", image_matches);
        waitKey(_step_time);
    }
    else
    {
        Mat image_matches;
        stringstream ss;
        ss<<_rgbPath<<frame_index<<".png";
        Mat rgb = imread( ss.str(), 0);
        drawMatches( rgb, kp1, _currRGB, kp2, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
        imshow("match", image_matches);
        waitKey(_step_time);
    }
    

    // 旋转向量转换成旋转矩阵
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

void GraphicEnd::saveFinalResult( string fileaddr )
{
    //补充闭环
    findMoreLoops();
    // g2o进行全局优化
    cout<<"saving final result"<<endl;
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    opt.setVerbose( true );
    opt.initializeOptimization();
    opt.optimize( atoi(g_pParaReader->GetPara("optimize_step").c_str() ) );

    stringstream ss;
    ofstream fout("./data/keyframe.txt");
    
    for (size_t i=0; i<_keyframes.size(); i++)
    {
        cout<<"keyframe "<<i<<" id = "<<_keyframes[i].id<<endl;
        fout<<_keyframes[i].id<<" "<<_keyframes[i].frame_index<<endl;
    }
    opt.save("./data/final_after.g2o");
    fout.close();
}

//回环检测：在过去的帧中随机取_loopclosure_frames那么多帧进行两两比较
void GraphicEnd::loopClosure()
{
    if (_keyframes.size() <= 3 )  //小于3时，回环没有意义
        return;
    cout<<"Checking loop closure."<<endl;
    vector<int> checked;
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;

    //相邻帧
    for (int i=-3; i>-5; i--)
    {
        int n = _keyframes.size() + i;
        if (n>=0)
        {
            vector<PLANE>& p1 = _keyframes[n].planes;
            RESULT_OF_MULTIPNP result = multiPnP( p1,
                                                  _currKF.planes, true, _keyframes[n].frame_index, _loop_closure_inliers );
            Eigen::Isometry3d T = result.T;
            if (T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
                continue;
            if ( result.norm > _loop_closure_error )
                continue;
            if (result.inliers < _loop_closure_inliers)
                continue;
            T = T.inverse();
            //若匹配上，则在两个帧之间加一条边
            EdgeSE3* edge = new EdgeSE3();
            edge->vertices() [0] = opt.vertex( _keyframes[n].id );
            edge->vertices() [1] = opt.vertex( _currKF.id );
            Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
            information(0, 0) = information(2,2) = 100; 
            information(1,1) = 100;
            information(3,3) = information(4,4) = information(5,5) = 100; 
            edge->setInformation( information );
            edge->setMeasurement( T );
            edge->setRobustKernel( _pSLAMEnd->_robustKernel );
            opt.addEdge( edge );
        }
        else
            break;
    }

    //随机搜索
    cout<<"checking random frames"<<endl;
    for (int i=0; i<_loopclosure_frames; i++)
    {
        int frame = rand() % (_keyframes.size() -3 ); //随机在过去的帧中取一帧
        if ( find(checked.begin(), checked.end(), frame) != checked.end() ) //之前已检查过
            continue;
        checked.push_back( frame );
        vector<PLANE>& p1 = _keyframes[frame].planes;
        RESULT_OF_MULTIPNP result = multiPnP( p1, _currKF.planes, true, _keyframes[frame].frame_index, _loop_closure_inliers );
        Eigen::Isometry3d T = result.T;
        
        if ( T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
            continue;
        if ( result.norm > _loop_closure_error )
            continue;
        if (result.inliers < _loop_closure_inliers)
            continue;
        T = T.inverse();
        displayLC( _keyframes[frame].frame_index, _currKF.frame_index, result.norm, result.inliers);
        //若匹配上，则在两个帧之间加一条边
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt.vertex( _keyframes[frame].id );
        edge->vertices() [1] = opt.vertex( _currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(2,2) = 100; 
        information(1,1) = 100;
        information(3,3) = information(4,4) = information(5,5) = 100; 
        edge->setInformation( information );
        edge->setMeasurement( T );
        edge->setRobustKernel( _pSLAMEnd->_robustKernel );
        opt.addEdge( edge );

        _keyframes.back().connect.push_back( frame );
    }
}

void GraphicEnd::lostRecovery()
{
    //以present作为新的关键帧
    cout<<BOLDYELLOW<<"Lost Recovery..."<<RESET<<endl;
    //把present中的数据存储到current中
    _currKF.id ++;
    _currKF.planes = _present.planes;
    _currKF.frame_index = _index;
    _lastRGB = _currRGB.clone();
    _kf_pos = _robot;

    ofstream fout("./data/lost.txt", ofstream::app);
    fout<<_currKF.id<<" "<<_currKF.frame_index<<endl;
    fout.close();
    
    _keyframes.push_back( _currKF );
    
    //将current关键帧存储至全局优化器
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    //顶点
    VertexSE3* v = new VertexSE3();
    v->setId( _currKF.id );
    if (_use_odometry)
        v->setEstimate( _odo_this );
    else
        v->setEstimate( Eigen::Isometry3d::Identity() );
    opt.addVertex( v );

    //由于当前位置不知道,所以不添加与上一帧相关的边
    //有里程计的时候，以里程计的读数为准
    if (_use_odometry)
    {
        Eigen::Isometry3d To = _odo_last.inverse()*_odo_this;
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices()[0] = opt.vertex( _currKF.id - 1 );
        edge->vertices()[1] = opt.vertex( _currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(2,2) = information(1, 1) = information(3,3) = information(4,4) = information(5,5) = 1/(_error_odometry*_error_odometry);
        edge->setInformation( information );
        edge->setMeasurement( To );
        opt.addEdge( edge );
        _odo_last = _odo_this;
        _lost = 0;
        return;
    }
    //check loop closure
    for (int i=0; i<_keyframes.size() - 1; i++)
    {
        vector<PLANE>& p1 = _keyframes[ i ].planes;
        RESULT_OF_MULTIPNP result = multiPnP( p1, _currKF.planes );
        Eigen::Isometry3d T = result.T;
        
        if (T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
            continue;
        if (result.inliers < _loop_closure_inliers )
            continue;
        if ( result.norm > _loop_closure_error )
            continue;
        T = T.inverse();
        //若匹配上，则在两个帧之间加一条边
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt.vertex( _keyframes[i].id );
        edge->vertices() [1] = opt.vertex( _currKF.id );
        Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
        information(0, 0) = information(1,1) = information(2,2) = 100; 
        information(3,3) = information(4,4) = information(5,5) = 100; 
        edge->setInformation( information );
        edge->setMeasurement( T );
        edge->setRobustKernel( _pSLAMEnd->_robustKernel );
        opt.addEdge( edge );

        _keyframes.back().connect.push_back( i );
    }
    
}

void GraphicEnd::displayLC( int frame1, int frame2, double norm, int inliers)
{
    static ofstream fout("./data/lc.txt");
    
    Mat rgb1, rgb2;
    stringstream ss;
    ss<<_rgbPath<<frame1<<".png";
    string fileaddr;
    ss>>fileaddr;
    ss.clear();
    
    rgb1 = imread( fileaddr, 0 );

    ss<<_rgbPath<<frame2<<".png";
    fileaddr.clear();
    ss>>fileaddr;
    rgb2 = imread( fileaddr, 0);

    imshow("loopClosure_1", rgb1);
    imshow("loopClosure_2", rgb2);

    fout<<frame1<<" "<<frame2<<" "<<norm<<" "<<inliers<<endl;
}

void GraphicEnd::findMoreLoops( )
{
    cout<<"Find more loops"<<endl;
    _moreLoops = 0;
    for (size_t i=0; i<_keyframes.size(); i++)
    {
        if (_keyframes[i].connect.size() == 0)
            continue;
        vector<int> checked; //检测出连上的之前的帧
        for (size_t j=0; j<_keyframes[i].connect.size(); j++)
        {
            //检测这个关键帧与connect之前与之后的帧的关系
            checked = checknearby( i, _keyframes[i].connect[j] );
            //然后检测这些帧与关键帧附近帧的关系
            for (size_t k=0; k<checked.size(); k++)
            {
                checknearby( checked[k], i );
            }
        }
    }
    cout<<BOLDRED<<"Total "<<_moreLoops<<" loops found. "<<RESET<<endl;
}

bool GraphicEnd::check( int frame1, int frame2 )
{
    cout<<YELLOW<<"checking "<<frame1<<", "<<frame2<<RESET<<endl;
    vector<PLANE>& p1 = _keyframes[frame1].planes;
    vector<PLANE>& p2 = _keyframes[frame2].planes;
    RESULT_OF_MULTIPNP result = multiPnP( p1, p2, true, _keyframes[frame1].frame_index, _loop_closure_inliers );
    Eigen::Isometry3d T = result.T;
    if (T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
        return false;
    if ( result.norm > _loop_closure_error )
        return false;
    if (result.inliers < _loop_closure_inliers)
        return false;
    T = T.inverse();
    EdgeSE3* edge = new EdgeSE3();
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    edge->vertices() [0] = opt.vertex( _keyframes[frame1].id );
    edge->vertices() [1] = opt.vertex( _keyframes[frame2].id );
    Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
    information(0, 0) = information(2,2) = 100; 
    information(1,1) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100; 
    edge->setInformation( information );
    edge->setMeasurement( T );
    edge->setRobustKernel( _pSLAMEnd->_robustKernel );
    opt.addEdge( edge );
    _moreLoops++;
    return true;
}

vector<int> GraphicEnd::checknearby( int source, int target )
{
    cout<<RED<<"checking "<<source<<" and "<<target<<RESET<<endl;

    vector<int> checked;
    int index = target;
    while( index>0 )
    {
        index--;
        if (index == source)
            continue;
        bool s = check( source, index);
        if (s == true)
            checked.push_back( index );
        else
            break;
    }
    index = target;
    while (index < (_keyframes.size()-1))
    {
        index++;
         if (index == source)
            continue;
        bool s = check( source, index);
        if (s == true)
            checked.push_back( index );
        else
            break;
    }
    return checked;
}
