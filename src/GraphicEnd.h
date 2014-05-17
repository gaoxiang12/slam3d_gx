#pragma once

#include <iostream>
#include <string>
#include "ImageReader.h"
#include "FeatureGrabber.h"
#include "FeatureManager.h"
#include "PCL_End.h"
#include <g2o/types/slam2d/se2.h>

using namespace g2o;
using namespace std;



class GraphicEnd
{

 public:
    GraphicEnd();
    ~GraphicEnd();

    int run();
    int run_once();

    void drawRobot(Mat& img);
    
 public:

    ImageReaderBase* pImageReader;
    FeatureGrabberBase* pFeatureGrabber;
    FeatureManager* pFeatureManager;
    FeatureManager2* pFeatureManager2;
    PCL_End* pPCLEnd;
    SE2 _robot_curr;      //当前机器人所在位置

    int _loops;
    bool _success;        //标记当前循环是否成功匹配到特征
    int _step_time;
    bool _need_global_optimization;
};

