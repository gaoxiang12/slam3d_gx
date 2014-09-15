#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

const double camera_factor = 1000;
const double camera_cx = 319.5;
const double camera_cy = 235.5;
const double camera_fx = 525.0;
const double camera_fy = 525.0;


void usage()
{
    cout<<"Usage: convert2pcd start_index end_index"<<endl;
}

int main( int argc, char** argv)
{
    if (argc != 3)
    {
        usage();
        return -1;
    }

    int start = atoi(argv[1]), end = atoi(argv[2]);

    Mat rgb, dep;
    char buffer[128];
    PointCloud::Ptr cloud( new PointCloud());

    for (int i = start; i<end; i++)
    {
        string str;
        sprintf( buffer, "./rgb_index/%d.png", i);
        str = buffer;
        rgb = imread( str, CV_LOAD_IMAGE_COLOR );
        sprintf( buffer, "./dep_index/%d.png", i);
        str = buffer;
        dep = imread( str, -1 );
        for (int m=0; m<dep.rows; m++)
        {
            for (int n=0; n<dep.cols; n++)
            {
                ushort d =  dep.ptr<ushort>(m) [n];
                if (d == 0)
                    continue;
                PointT p;
                uchar* data = rgb.ptr<uchar>(m);
                uint8_t b = data[n*rgb.channels()], g = data[n*rgb.channels()+1],  r = data[n*rgb.channels()+2];
                //p.r = r, p.g = g, p.b = b;
                p.rgba = ((int)r) << 16 | ((int)g) << 8 | ((int)b);
                double z = double(d)/camera_factor;
                double x = ( n - camera_cx) * z / camera_fx;
                double y = ( m - camera_cy) * z / camera_fy;
                p.x = x; p.y = y; p.z = z;
                cloud->points.push_back( p );
            }
        }
        sprintf( buffer, "./pcd/%d.pcd", i);
        str = buffer;
        cloud->height = 1;
        cloud->width = cloud->points.size();
        cloud->is_dense = false;
        pcl::io::savePCDFile(str, *cloud );
        cloud->points.clear();
        cout<<i<<" of "<<end-start<<" files ok. "<<endl;
    }

    return 0;
}
