/*****************************************
 * view loop closure
 *****************************************/
#include "const.h"
#include "GraphicEnd.h"
#include "ParameterReader.h"

#include <fstream>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

void usage()
{
    cout<<"viewLC lc.txt"<<endl;
}

int main( int argc, char** argv )
{
    if (argc != 2)
    {
        usage();
        return -1;
    }

    ifstream fin(argv[1]);
    if (!fin)
    {
        cout<<"file does not exist"<<endl;
        return -1;
    }

    g_pParaReader = new ParameterReader( "./parameters.yaml" );
    string data_source = g_pParaReader->GetPara( "data_source" );
    stringstream ss;
    
    while( !fin.eof() )
    {
        string fileaddr;
        int frame1, frame2;
        double norm;
        fin>>frame1>>frame2>>norm;
        Mat img1, img2;
        ss<<data_source<<"/rgb_index/"<<frame1<<".png";
        ss>>fileaddr;
        img1 = imread( fileaddr, 0);
        fileaddr.clear();
        ss.clear();
        ss<<data_source<<"/rgb_index/"<<frame2<<".png";
        ss>>fileaddr;
        img2 = imread( fileaddr, 0);
        ss.clear();

        imshow("lc1", img1);
        imshow("lc2", img2);
        cout<<"norm = "<<norm<<endl;
        waitKey(0);
    }

    return 0;
}

