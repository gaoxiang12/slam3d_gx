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
    _min_error_plane = atof( g_pParaReader->GetPara("min_error_plane").c_str() );
    _match_min_dist = atof( g_pParaReader->GetPara("match_min_dist").c_str() );
    _percent = atof( g_pParaReader->GetPara("plane_percent").c_str() );
    _max_pos_change = atof( g_pParaReader->GetPara("max_pos_change").c_str());
    _max_planes = atoi( g_pParaReader->GetPara("max_planes").c_str() );
    cout<<"max planes = "<<_max_planes<<endl;
    _loopclosure_frames = atoi( g_pParaReader->GetPara("loopclosure_frames").c_str() );
    _loop_closure_detection = (g_pParaReader->GetPara("loop_closure_detection") == string("yes"))?true:false;
    _loop_closure_error = atof(g_pParaReader->GetPara("loop_closure_error").c_str());
    _lost_frames = atoi( g_pParaReader->GetPara("lost_frames").c_str() );
    _robot = _kf_pos = Eigen::Isometry3d::Identity();
    
    //读取首帧图像并处理
    readimage();
    _lastRGB = _currRGB.clone();
    _currKF.id = 0;
    _currKF.frame_index = _index;
    _currKF.planes = extractPlanes( _currCloud ); //抓平面
    generateImageOnPlane( _currRGB, _currKF.planes, _currDep);
    for ( size_t i=0; i<_currKF.planes.size(); i++ )
    {
        _currKF.planes[i].kp = extractKeypoints(_currKF.planes[i].image);
        _currKF.planes[i].desp = extractDescriptor( _currRGB, _currKF.planes[i].kp );
        compute3dPosition( _currKF.planes[i], _currDep );
    }
    _keyframes.push_back( _currKF );
    //将current存储至全局优化器
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    VertexSE3* v = new VertexSE3();
    v->setId( _currKF.id );
    v->setEstimate( _robot );
    v->setFixed( true );
    opt.addVertex( v );
    _index ++;

    cout<<"********************"<<endl;
}

int GraphicEnd::run()
{
    //清空present并读取新的数据
    cout<<"********************"<<endl;
    _present.planes.clear();
    
    readimage();
    
    //处理present
    _present.planes = extractPlanes( _currCloud );
    generateImageOnPlane( _currRGB, _present.planes, _currDep );

    for ( size_t i=0; i<_present.planes.size(); i++ )
    {
        _present.planes[i].kp = extractKeypoints( _present.planes[i].image );
        _present.planes[i].desp = extractDescriptor( _currRGB, _present.planes[i].kp );
        compute3dPosition( _present.planes[i], _currDep);
    }

    // 求解present到current的变换矩阵
    Eigen::Isometry3d T = multiPnP( _currKF.planes, _present.planes );
    T = T.inverse();  //好像是反着的
    
    // 如果平移和旋转超过一个阈值，则定义新的关键帧
    Eigen::Vector3d rpy = T.rotation().eulerAngles(0, 1, 2);
    Eigen::Vector3d trans = T.translation();
    double norm = rpy.norm() + trans.norm()/10;
    cout<<RED<<"norm of T = "<<norm<<RESET<<endl;
    if ( T.matrix() == Eigen::Isometry3d::Identity().matrix() )
    {
        //未匹配上
        cout<<BOLDRED"This frame lost"<<RESET<<endl;
        _lost++;
    }
    else if (norm > 1.0)
    {
        //PnP计算结果不对
        cerr<<"an error occured."<<endl;
        _lost++;
        //waitKey(0);
    }
    else if (norm > _max_pos_change)
    {
        //生成一个新的关键帧
        _robot = T * _kf_pos;
        generateKeyFrame(T);
        if (_loop_closure_detection == true)
            loopClosure();
        _lost = 0;
    }
    else
    {
        //小变化，更新robot位置
        _robot = T * _kf_pos;
        _lost = 0;
    }
    if (_lost > _lost_frames)
    {
        cerr<<"the robot lost. Perform lost recovery."<<endl;
        //waitKey(0);
        lostRecovery();
    }


    cout<<RED<<"key frame size = "<<_keyframes.size()<<RESET<<endl;
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
    ss.str("");
    ss.clear();
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

    cout<<"add key frame: "<<_currKF.id<<endl;
    //waitKey(0);
    _keyframes.push_back( _currKF );
    
    //将current关键帧存储至全局优化器
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    //顶点
    VertexSE3* v = new VertexSE3();
    v->setId( _currKF.id );
    //v->setEstimate( _robot );
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
}

vector<PLANE> GraphicEnd::extractPlanes( PointCloud::Ptr cloud)
{
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

        if (i == _max_planes)
            break;
    }

    cout<<"Total planes: "<<i<<endl;
    return planes;
}

void GraphicEnd::generateImageOnPlane( Mat rgb, vector<PLANE>& planes, Mat depth)
{
    cout<<"GraphicEnd::generateImageOnPlane"<<endl;

    int rows = rgb.rows;
    int cols = rgb.cols;
    clock_t start, end;
    start = clock();
    for (size_t t = 0; t<planes.size(); t++)
    {
        planes[t].image = rgb.clone();
    }
    end = clock();
    double dur = (double)(end - start);
    cout<<"clone time = "<<(dur/CLOCKS_PER_SEC);

    start = clock();
    int m = rows/10;
    int n = cols/10;
    for (int j=0; j<m; j++)
    {
        uchar* data = rgb.ptr<uchar> (j*10+5); //行指针
        ushort* depData = depth.ptr<ushort> (j*10+5);
        for (int i=0; i<n; i++)
        {
            ushort d = depData[i*10+5];
            if (d == 0)
            {
                for (size_t t=0; t<planes.size(); t++)
                {
                    for (int k=0; k<10; k++)
                        for (int l=0; l<10; l++)
                            planes[t].image.ptr<uchar> (j*10+k) [i*10+l] = 0;
                }
                continue;
            }
            double z = double(d)/camera_factor;
            double x = ( i*10+5 - camera_cx) * z / camera_fx;
            double y = ( j*10+5- camera_cy) * z / camera_fy;

            for (size_t t=0; t<planes.size(); t++)
            {
                pcl::ModelCoefficients coff = planes[t].coff;
                double e = coff.values[0]*x + coff.values[1]*y + coff.values[2]*z + coff.values[3];
                e*= e;
                //cout<<e<<endl;

                if (e > _min_error_plane)
                {
                    for (int k=0; k<10; k++)
                        for (int l=0; l<10; l++)
                            planes[t].image.ptr<uchar> (j*10+k) [i*10+l] = 0;
                }
            }
        }
    }

    end = clock();
    dur = (double)(end - start);
    cout<<"computation time = "<<(dur/CLOCKS_PER_SEC);

    start = clock();
    
    for (size_t t=0; t<planes.size(); t++)
    {
        Mat dst;
        equalizeHist( planes[t].image, dst );
        planes[t].image = dst.clone();
        //cout<<"showing image on plane"<<endl;
        //imshow("Planes", planes[t].image);
        //waitKey( _step_time );
    }
    end = clock();
    dur = (double)(end - start);
    cout<<"equalizehist time = "<<(dur/CLOCKS_PER_SEC);

}

void GraphicEnd::compute3dPosition( PLANE& plane, Mat depth)
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
    double max_dist = 0, min_dist = 100;
    for (int i=0; i<des1.rows; i++)
    {
        double dist = matches[ i ].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
    cout<<"plane match: total "<<matches.size()<<" matches. min dist = "<<min_dist<<endl;
    //choose good matches
    vector<DMatch> good_matches;
    
    for (int i=0; i<des1.rows; i++)
    {
        //cout<<matches[i].distance<<endl;
        if (matches[ i ].distance <= max(4*min_dist, _match_min_dist))
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

vector<DMatch> GraphicEnd::pnp( PLANE& p1, PLANE& p2)
{
    vector<DMatch> matches = match( p1.desp, p2.desp );
    cout<<"good matches: "<<matches.size()<<endl;
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
    //cout<<"calling solvePnPRansac"<<endl;
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);
    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( matches[inliers.at<int>(i,0)] );
    
    cout<<"PnP inliers = "<<inliers.rows<<endl;

    Mat image_matches;
    drawMatches(p1.image, p1.kp, p2.image, p2.kp, inlierMatches, image_matches, Scalar::all(-1), CV_RGB(255,255,255), Mat(), 4);
    imshow("match_of_planes", image_matches);
    waitKey(_step_time);

    return inlierMatches;
}

//通过若干个平面求解PnP问题，当匹配不上时会返回I
Eigen::Isometry3d GraphicEnd::multiPnP( vector<PLANE>& plane1, vector<PLANE>& plane2, bool loopclosure, int frame_index)
{
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
        return Eigen::Isometry3d::Identity();
    }
    
    double camera_matrix[3][3] = { { camera_fx, 0, camera_cx }, { 0, camera_fy ,camera_cy }, { 0, 0, 1 }};
    Mat cameraMatrix(3,3,CV_64F, camera_matrix);
    Mat rvec, tvec; 
    Mat inliers;
    solvePnPRansac(obj, img, cameraMatrix, Mat(), rvec, tvec, false, 100, 8.0, 100, inliers);

    if (inliers.rows < 12)
        return Eigen::Isometry3d::Identity();

    vector<DMatch> inlierMatches;
    for (int i=0; i<inliers.rows; i++)
        inlierMatches.push_back( match_show[inliers.at<int>(i,0)] );
    
    cout<<CYAN<<"multiICP::inliers = "<<inliers.rows<<RESET<<endl;
    cout<<"matches: "<<match_show.size()<<endl;
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

    /*
    vector<Eigen::Isometry3d> transforms;
    for (size_t i=0; i<matches.size(); i++)
    {
        cout<<"planes 1:"<<matches[i].queryIdx<<", planes 2:"<<matches[i].trainIdx<<endl;
        Eigen::Isometry3d t = pnp( plane1[matches[i].queryIdx], plane2[matches[i].trainIdx] );
        cout<<"T"<<i<<" = "<<endl;
        cout<<t.matrix()<<endl;
        transforms.push_back(t);
    }

    if (transforms.empty())
        return Eigen::Isometry3d::Identity();
    if (transforms.size() == 1)
    {
        return transforms[0];
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

    */
    //return esti;
}

void GraphicEnd::saveFinalResult( string fileaddr )
{
    // g2o进行全局优化
    cout<<"saving final result"<<endl;
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    opt.setVerbose( true );
    opt.initializeOptimization();
    opt.optimize( atoi(g_pParaReader->GetPara("optimize_step").c_str() ) );

    // 拼合所有关键帧
    PointCloud::Ptr output(new PointCloud());
    PointCloud::Ptr curr( new PointCloud());
    stringstream ss;
    pcl::VoxelGrid<PointT> voxel;
    double grid_leaf = atof(g_pParaReader->GetPara("grid_leaf").c_str() );
    voxel.setLeafSize( grid_leaf, grid_leaf, grid_leaf );

    ofstream fout("./data/keyframe.txt");
    
    for (size_t i=0; i<_keyframes.size(); i++)
    {
        cout<<"keyframe "<<i<<" id = "<<_keyframes[i].id<<endl;
        fout<<_keyframes[i].id<<" "<<_keyframes[i].frame_index<<endl;
        /*
        ss<<_pclPath<<_keyframes[i].frame_index<<".pcd";
        string str;
        ss>>str;
        cout<<"loading "<<str<<endl;
        ss.clear();
        pcl::io::loadPCDFile( str, *curr );

        VertexSE3* pv = dynamic_cast<VertexSE3*> (opt.vertex( _keyframes[i].id ));
        if (pv == NULL)
            break;
        Eigen::Isometry3d pos = pv->estimate();
        Eigen::Vector3d rpy = pos.rotation().eulerAngles(0, 1, 2);
        Eigen::Vector3d trans = pos.translation();
        double norm = rpy.norm() + trans.norm();
        if (norm >= 100)
        {
            continue;
        }
        
        cout<<"Node "<<i<<" T = "<<endl;
        cout<<pos.matrix()<<endl;
        
        PointCloud::Ptr tmp( new PointCloud());
        pcl::transformPointCloud( *curr, *tmp, pos.matrix());
        *output += *tmp;

        //格点滤波
        voxel.setInputCloud( output );
        PointCloud::Ptr output_filtered( new PointCloud );
        voxel.filter( *output_filtered );
        output->swap( *output_filtered );

        */
    }
    //存储点云
    //pcl::io::savePCDFile( fileaddr, *output );
    //cout<<"final result is saved at "<<fileaddr<<endl;
    opt.save("./data/final_after.g2o");
    fout.close();
}

//回环检测：在过去的帧中随机取_loopclosure_frames那么多帧进行两两比较
void GraphicEnd::loopClosure()
{
    if (_keyframes.size() <= 3 )  //小于3时，回环没有意义
        return;
    cout<<"Checking loop closure."<<endl;
    waitKey(10);
    vector<int> checked;
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;

    //相邻帧
    for (int i=-2; i>-5; i--)
    {
        int n = _keyframes.size() + i;
        if (n>=0)
        {
            vector<PLANE>& p1 = _keyframes[n].planes;
            Eigen::Isometry3d T = multiPnP( p1, _currKF.planes );
            if (T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
                continue;
            Eigen::Vector3d rpy = T.rotation().eulerAngles(0, 1, 2);
            Eigen::Vector3d trans = T.translation();
            double norm = rpy.norm() + trans.norm();
            if (norm > _loop_closure_error) //匹配错误
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
    //搜索种子帧
    cout<<"checking seeds, seed.size()"<<_seed.size()<<endl;
    vector<int> newseed;
    for (size_t i=0; i<_seed.size(); i++)
    {
        vector<PLANE>& p1 = _keyframes[_seed[i]].planes;
        Eigen::Isometry3d T = multiPnP( p1, _currKF.planes );
        if (T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
            continue;
        Eigen::Vector3d rpy = T.rotation().eulerAngles(0, 1, 2);
        Eigen::Vector3d trans = T.translation();
        double norm = rpy.norm() + trans.norm();
        cout<<"norm = "<<norm<<endl;
        if (norm > _loop_closure_error) //匹配错误
            continue;
        T = T.inverse();
        //若匹配上，则在两个帧之间加一条边
        checked.push_back( _seed[i] );
        newseed.push_back( _seed[i] );
        EdgeSE3* edge = new EdgeSE3();
        edge->vertices() [0] = opt.vertex( _keyframes[_seed[i]].id );
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

    //随机搜索
    cout<<"checking random frames"<<endl;
    for (int i=0; i<_loopclosure_frames; i++)
    {
        int frame = rand() % (_keyframes.size() -3 ); //随机在过去的帧中取一帧
        if ( find(checked.begin(), checked.end(), frame) != checked.end() ) //之前已检查过
            continue;
        checked.push_back( frame );
        vector<PLANE>& p1 = _keyframes[frame].planes;
        Eigen::Isometry3d T = multiPnP( p1, _currKF.planes, true, _keyframes[frame].frame_index );
        
        if (T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
            continue;
        Eigen::Vector3d rpy = T.rotation().eulerAngles(0, 1, 2);
        Eigen::Vector3d trans = T.translation();
        double norm = rpy.norm() + trans.norm();
        cout<<"norm = "<<norm<<endl;
        if (norm > _loop_closure_error)
            continue;
        cout<<"T = "<<T.matrix()<<endl;
        T = T.inverse();
        
        displayLC( _keyframes[frame].frame_index, _currKF.frame_index, norm);
        newseed.push_back( frame );
        
        //若匹配上，则在两个帧之间加一条边
        cout<<BOLDBLUE<<"find a loop closure between kf "<<_currKF.id<<" and kf "<<frame<<RESET<<endl;
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
    }

    waitKey(10);
    _seed = newseed;
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

    //waitKey(0);
    _keyframes.push_back( _currKF );
    
    //将current关键帧存储至全局优化器
    SparseOptimizer& opt = _pSLAMEnd->globalOptimizer;
    //顶点
    VertexSE3* v = new VertexSE3();
    v->setId( _currKF.id );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    opt.addVertex( v );

    //由于当前位置不知道,所以不添加与上一帧相关的边
    //边
    EdgeSE3* edge = new EdgeSE3();
    edge->vertices()[0] = opt.vertex( _currKF.id - 1 );
    edge->vertices()[1] = opt.vertex( _currKF.id );
    Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
    information(0, 0) = information(1,1) = information(2,2) = 100; 
    information(3,3) = information(4,4) = information(5,5) = 100; 
    edge->setInformation( information );
    edge->setMeasurement( Eigen::Isometry3d::Identity() );
    opt.addEdge( edge );
    //check loop closure
    for (int i=0; i<_keyframes.size() - 1; i++)
    {
        vector<PLANE>& p1 = _keyframes[ i ].planes;
        Eigen::Isometry3d T = multiPnP( p1, _currKF.planes );
        
        if (T.matrix() == Eigen::Isometry3d::Identity().matrix()) //匹配不上
            continue;
        Eigen::Vector3d rpy = T.rotation().eulerAngles(0, 1, 2);
        Eigen::Vector3d trans = T.translation();
        double norm = rpy.norm() + trans.norm();
        if (norm > _loop_closure_error)
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
    }
    
    
    /*
    //边
    EdgeSE3* edge = new EdgeSE3();
    edge->vertices()[0] = opt.vertex( _currKF.id - 1 );
    edge->vertices()[1] = opt.vertex( _currKF.id );
    Matrix<double, 6,6> information = Matrix<double, 6, 6>::Identity();
    information(0, 0) = information(1,1) = information(2,2) = 100; 
    information(3,3) = information(4,4) = information(5,5) = 100; 
    edge->setInformation( information );
    edge->setMeasurement( T );
    opt.addEdge( edge );
    */
}

void GraphicEnd::displayLC( int frame1, int frame2, double norm)
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
    waitKey(10);

    fout<<frame1<<" "<<frame2<<" "<<norm<<endl;
}
