#include "GraphicEnd.h"
#include "const.h"
#include "FeatureGrabber.h"
#include "ParameterReader.h"
#include "FABMAP_End.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

GraphicEnd::GraphicEnd()
{
    cout<<"construct graphic end"<<endl;
    g_pParaReader = new ParameterReader(parameter_file_addr);
    if (g_pParaReader->GetPara("fabmap") == string("yes"))
    {
        g_pFAB = new FABMAP_End();
    }
    else
        g_pFAB = NULL;
    
    pImageReader = new ImageReader(parameter_file_addr);
    pFeatureGrabber = new FeatureGrabber(
                                         g_pParaReader->GetPara("detector_name"),
                                         g_pParaReader->GetPara("descriptor_name"));

    pFeatureManager = new FeatureManager(atoi(g_pParaReader->GetPara("save_if_seen").c_str()),
                                         pFeatureGrabber,
                                         -atoi(g_pParaReader->GetPara("del_not_seen").c_str()));

    pFeatureManager2 = new FeatureManager2( pFeatureGrabber,
                                            pImageReader,
                                            pFeatureManager);
    
    cout<<"set up pcl end"<<endl;
    pPCLEnd = new PCL_End(g_pParaReader->GetPara("data_source"));
    
    if (vision == true)
    {
        namedWindow("slam_gx");
    }

    _step_time = atoi(g_pParaReader->GetPara("step_time").c_str());
    _loops = 0;
}

GraphicEnd::~GraphicEnd()
{
    delete pImageReader;
    delete pFeatureGrabber;
    delete pFeatureManager;
    delete pFeatureManager2;
    delete pPCLEnd;
    
    delete g_pParaReader;
    if (g_pFAB)
        delete g_pFAB;
}

int GraphicEnd::run()
{
    return 1;
}

int GraphicEnd::run_once()
{
    _need_global_optimization = false;
    _success = false;
    stringstream ss;
    string logfile;

    ofstream fout;
    static ofstream frobot("log/robot.path");
    frobot<<_loops<<": "<<_robot_curr[0]<<", "<<_robot_curr[1]<<", "<<_robot_curr[2]<<endl;
    
    ss<<"log/log_"<<_loops<<".txt";
    ss>>logfile;
    fout.open(logfile.c_str());
    
    if (debug_info)
    {
        cout<<"\n-- GraphicEnd::loop "<<_loops<<" --"<<endl;
        cout<<"graphic end : robot in on "<<_robot_curr[0]<<", "<<_robot_curr[1]<<", rotation = "<<_robot_curr[2]<<endl;
    }
    if (pImageReader->Next() == 0) //存在下一张图片
        return 0;
    //读取rgb图与深度图    
    Mat rgb = pImageReader->GetRGB();
    Mat dep = pImageReader->GetDep();
    
    pFeatureGrabber->SetRGBDep(rgb, dep);

    //抓取当前图形的特征点与描述子
    vector<KeyPoint> keyPoints = pFeatureGrabber->GetKeyPoints();
    Mat desc = pFeatureGrabber->GetDescriptors();

    //将当前图像的特征与机器人位置传送至特征数据库
    //pFeatureManager->Input(keyPoints, desc, _robot_curr);
    try {
        pFeatureManager2->Input(keyPoints, desc, _robot_curr, _loops);
    } catch (GRAPHIC_END_NEED_GLOBAL_OPTIMIZATION e) {
        e.disp();
        _need_global_optimization = true;
        if (g_pFAB)
            g_pFAB->process( pImageReader->GetCurrentFileName(), _loops );
        if (g_pParaReader->GetPara("online_pcl") == string("yes"))
        {
            pPCLEnd->addInput(pImageReader->GetCurrentIndex(), _robot_curr);
            pPCLEnd->save();
        }
        ss.clear();
        string s;
        ss<<"keyframe/"<<pImageReader->GetCurrentIndex()<<".bmp";
        ss>>s;
        cout<<"save to "<<s<<endl;
        imwrite(s, rgb);
    }
    
    if (vision == true)
    {
        Mat image_with_keypoints;
        drawKeypoints(rgb, keyPoints, image_with_keypoints);
        drawRobot(image_with_keypoints);
        imshow("slam_gx", image_with_keypoints);
        
        waitKey(_step_time);
    }
    fout.close();

    _loops++;
    return 1;
}

void GraphicEnd::drawRobot(Mat& img)
{
    rectangle( img, Point2f(400, 350), Point2f(500, 450), Scalar(255,0,0), 3 );
    
    Point2f p(450-_robot_curr[1]*20, 400-_robot_curr[0]*20);
    //位置信息
    circle(img, Point2f(450,400), 2, Scalar(0, 0, 255), 2);
    circle(img, p, 10, Scalar(255, 0, 0), 1);

    //转角信息
    double alpha = -PI/2-_robot_curr[2];
    Point2f p2( p.x + 20*cos(alpha), p.y + 20 * sin(alpha) );
    line(img, p, p2, Scalar(255,0,0), 1);
}
