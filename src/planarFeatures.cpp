#include <iostream>
#include <vector>

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>

const double camera_factor = 1000.0;
const double camera_cx = 320.0, camera_cy = 235.5, camera_fx = 525.0, camera_fy = 525.0;

using namespace std;
using namespace cv;

void usage()
{
    cout<<"planarFeatures rgb dep "<<endl;
}

bool isPlanar( KeyPoint kp, Mat& dep, Mat& rgb);

int main( int argc, char** argv)
{
    if (argc < 3)
    {
        usage();
        return -1;
    }
    double threshold = 5;
    if (argc == 4)
    {
        threshold = atof( argv[3] );
    }
    
    Mat rgb, dep;
    rgb = imread( argv[1], 0);
    dep = imread( argv[2], -1);

    imshow("rgb", rgb);
    imshow("dep", dep);
    cout<<threshold<<endl;
    vector<KeyPoint> kp;
    FAST(rgb, kp, threshold );
    Mat img_kp;
    drawKeypoints(rgb, kp, img_kp, Scalar::all(-1), 0);
    imshow("original key points", img_kp);
    waitKey(0);

    vector<KeyPoint> kp_valid;
    for (size_t i=0 ; i<kp.size(); i++)
    {
        double u = kp[i].pt.x, v = kp[i].pt.y;
        unsigned short d = dep.at<unsigned short>(round(v), round(u));
        if (d == 0)
        {
            continue;
        }
        kp_valid.push_back( kp[i] );
    }

    drawKeypoints(rgb, kp_valid, img_kp, Scalar::all(-1), 0);
    imshow("valid key points", img_kp);

    /*
    Mat img_kp_on_dep;
    drawKeypoints(dep, kp_valid, img_kp_on_dep, Scalar::all(-1), 0);
    imshow("keypoint on dep", img_kp_on_dep );
    */
    waitKey(0);

    vector<KeyPoint> kp_planar;
    for (size_t i = 0; i<kp_valid.size(); i++)
    {
        if ( isPlanar(kp_valid[i], dep, rgb) )
            kp_planar.push_back( kp_valid[i] );
    }
    cout<<"total kp: "<<kp.size()<<", valid: "<<kp_valid.size()<<", planar: "<<kp_planar.size();
    drawKeypoints(rgb, kp_planar, img_kp, Scalar::all(-1), 0);
    imshow("planar key points", img_kp);
    waitKey(0);
    return 0;
}

bool isPlanar( KeyPoint kp, Mat& dep, Mat& rgb)
{
    int u = (int)kp.pt.x;
    int v = (int)kp.pt.y;
    Mat near = dep( Range(v-3, v+4), Range(u-3, u+4));
    //cout<<near<<endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ> ());
    Mat result = rgb.clone();
    circle(result, kp.pt, 10, Scalar(255, 0, 0));
    imshow("planar checking...", result);
    waitKey(1000);
    for (int j=0; j<7; j++)
        for (int i=0; i<7; i++)
        {
            if (near.ptr<ushort> (j) [i] == 0)
            {
                cout<<"has zero."<<endl;
                return false;
            }
            double d = (double)near.ptr<ushort> (j) [i];
            double z = double(d)/camera_factor;
            double x = ( u +(i-3)- camera_cx) * z / camera_fx;
            double y = ( v +(j-3) - camera_cy) * z / camera_fy;
            cout<<"xyz = "<<x<<","<<y<<","<<z<<endl;
            cloud->push_back( pcl::PointXYZ( x,y,z ));
        }
    //cout<<"u="<<u<<", v="<<v<<endl;
    //cout<<near<<endl;

    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane( new
                                                                    pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud)
                                                                    );
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac( model_plane );
    ransac.setDistanceThreshold( 0.01);
    ransac.computeModel();
    vector<int> inliers;
    ransac.getInliers( inliers );
    
    cout<<"inlier size = "<<inliers.size()<<endl;
    if (inliers.size() > 40 )
    {
        cout<<"is planar."<<endl;
        return true;
    }

    cout<<"no enough inliers"<<endl;
    return false;
}

