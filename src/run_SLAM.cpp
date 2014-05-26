#include "GraphicEnd.h"

#include <iostream>
using namespace std;

void usage()
{
    cout<<"usage: run_SLAM loops"<<endl;
}

int main( int argc, char** argv )
{
    int nloops;

    if (argc < 2)
    {
        usage();
        nloops = 3;
    }

    GraphicEnd* pGraphicEnd = new GraphicEnd();
    SLAMEnd* pSLAMEnd = new SLAMEnd();

    pGraphicEnd->init( pSLAMEnd );
    pSLAMEnd->init( pGraphicEnd );

    if (argc == 2)
        nloops = atoi( argv[1] );
    
    for (int i=0; i< nloops; i++)
    {
        pGraphicEnd->run();
    }

    cout<<"Total KeyFrame: "<<pGraphicEnd->_keyframes.size()<<endl;
    
    delete pGraphicEnd;
    delete pSLAMEnd;

    return 0;
}
