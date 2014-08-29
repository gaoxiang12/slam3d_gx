#include "const.h"
#include "ParameterReader.h"
#include "GraphicEnd.h"

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <fstream>

using namespace std;

void usage()
{
    cout<<"saveOutput keyframe.txt final.g2o [ pass_z ]"<<endl;
}

int main( int argc, char ** argv)
{
    if (argc < 3)
    {
        usage();
        return -1;
    }
    
    g_pParaReader = new ParameterReader( parameter_file_addr );
    SLAMEnd slam;
    slam.init(NULL);
    SparseOptimizer& opt = slam.globalOptimizer;
    opt.load(argv[2]);
    ifstream fin(argv[1]);
    PointCloud::Ptr output(new PointCloud());
    PointCloud::Ptr curr( new PointCloud());
    stringstream ss;
    pcl::VoxelGrid<PointT> voxel;
    double grid_leaf = atof(g_pParaReader->GetPara("grid_leaf").c_str() );
    voxel.setLeafSize( grid_leaf, grid_leaf, grid_leaf );
    string _pclPath = g_pParaReader->GetPara("data_source")+"/pcd/";

    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    double z = 5.0;
    if (argc == 4)
        z = atof( argv[ 3 ]);
    pass.setFilterLimits(0.0, z);
    
    while( !fin.eof() )
    {
        int frame, id;
        fin>>id>>frame;
        ss<<_pclPath<<frame<<".pcd";
        
        string str;
        ss>>str;
        cout<<"loading "<<str<<endl;
        ss.clear();

        pcl::io::loadPCDFile( str, *curr );

        VertexSE3* pv = dynamic_cast<VertexSE3*> (opt.vertex( id ));
        if (pv == NULL)
        {
            cout<<"cannot find vertex: "<<id<<endl;
            continue;
        }
        Eigen::Isometry3d pos = pv->estimate();
        /*
        Eigen::Vector3d rpy = pos.rotation().eulerAngles(0, 1, 2);
        Eigen::Vector3d trans = pos.translation();
        double norm = rpy.norm() + trans.norm();
        if (norm >= 100)
        {
            continue;
        }
        */
        
        cout<<pos.matrix()<<endl;
        voxel.setInputCloud( curr );
        PointCloud::Ptr tmp( new PointCloud());
        voxel.filter( *tmp );
        curr.swap( tmp );
        //z方向滤波
        pass.setInputCloud( curr );
        pass.filter(*tmp);
        curr->swap( *tmp );
        
        pcl::transformPointCloud( *curr, *tmp, pos.matrix());
        *output += *tmp;

    }
    voxel.setInputCloud( output );
    PointCloud::Ptr output_filtered( new PointCloud );
    voxel.filter( *output_filtered );
    output->swap( *output_filtered );
    pcl::io::savePCDFile( "result.pcd", *output);
    cout<<"final result saved."<<endl;
    delete g_pParaReader;
}
