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
    stringstream ss;
    ss<<g_pParaReader->GetPara("data_source")<<"/associate.txt";
    ss>>fileaddr;
    ss.clear();
    ifstream asso( fileaddr.c_str() );

    ofstream fout("trajectory.txt");
    string init_time;
    double init_data[7];

    ss<<g_pParaReader->GetPara("data_source")<<"/groundtruth.txt";
    ss>>fileaddr;
    ifstream groundtruth( fileaddr.c_str() );
    char buffer[100];
    for (int i=0; i<3; i++)
        groundtruth.getline(buffer, 100);

    groundtruth>>init_time;
    for (int i=0; i<7; i++)
        groundtruth>>init_data[i];
    
    int jump = 0;
    while (!fin.eof())
    {
        int frame, id;
        fin>>id>>frame;
        //读取associate.txt中相应的时间
        char buffer[100];
        for (int i=0; i<frame-jump; i++)
            asso.getline(buffer, 100);
        string time; //时间
        asso>>time;
        //位置
        VertexSE3* pv = dynamic_cast<VertexSE3*> (opt.vertex( id) );
        double data[7];
        pv->getEstimateData( data );

        fout<<time<<" ";
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

