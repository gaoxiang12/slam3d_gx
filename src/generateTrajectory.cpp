#include "const.h"
#include "ParameterReader.h"
#include "GraphicEnd.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

int usage()
{
    cout<<"generateTrajectory keyframe.txt final.g2o"<<endl;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        usage();
        return -1;
    }
    g_pParaReader = new ParameterReader( parameter_file_addr );
    SLAMEnd slam;
    slam.init( NULL );

    SparseOptimizer& opt = slam.globalOptimizer;
    opt.load(argv[2]);

    ifstream fin(argv[1]);
    if (!fin)
    {
        cout<<"file does not exist"<<endl;
        return -1;
    }

    string fileaddr;
    ifstream asso( string(g_pParaReader->GetPara("data_source")+string("/associate.txt")).c_str());
    stringstream ss;
    ofstream fout("trajectory.txt");
    string init_time;
    double init_data[7];
    /*
    asso>>timestamp;
    for (int i=0; i<7; i++)
        asso>>init_data[i];
    VertexSE3 v;
    v.setEstimateData( data );
    */
    //    Eigen::Isometry3d Ti = v.estimate();
    int jump = 1;
    while (!fin.eof())
    {
        int frame, id;
        fin>>id>>frame;
        //将asso跳若干行
        string timestamp;
        for (int i=0; i<frame-jump; i++)
            getline( asso, timestamp );
        asso>>timestamp;
        //位置
        VertexSE3* pv = dynamic_cast<VertexSE3*> (opt.vertex( id) );
        if (pv == NULL)
            continue;
        double data[7];
        pv->getEstimateData( data );
        fout<<timestamp<<" ";
        for (int i=0; i<7; i++)
            fout<<data[i]<<" ";
        fout<<endl;

        jump = frame;
    }
    cout<<"trajectory saved."<<endl;
    fout.close();
    fin.close();
    asso.close();

    delete g_pParaReader;
}

