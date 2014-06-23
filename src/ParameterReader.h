/*****************************************
 * ParameterReader.h 参数管理类
 * 程序中将有一个该类的全局变量，从parameter.yaml中读取各个参数的值，供其他类使用，方便调试
 ****************************************/

#pragma once
#include "const.h"
#include <string>
#include <sstream>
#include <iostream>

using namespace std;

class ParameterReader
{
 public:
    ParameterReader(const string& para_file);
    
    string GetPara(const string& para_name);

 protected:
    string num2string(double d)
    {
        ss.str("");
        ss.clear();
        ss<<d;
        return ss.str();
    }

    string num2string(int d)
    {
        ss.str("");
        ss.clear();
        ss<<d;
        return ss.str();
    }
    
 protected:
    stringstream ss;
    // configure parameters
    string _data_source; //数据来源
    string _detector_name;//特征相关：检测子名称
    string _descriptor_name; //描述子名称
    int _start_index;    //起始索引
    int _end_index;      //终止索引

    int _step_time;      //调试时每一次循环的等待时间
    double _error_threshold;  //错误阈值：标识相邻两帧位置不能相差太大

    //图优化参数
    int _optimize_step; //优化步数
    string _robust_kernel;

    //特征点相关参数
    double _match_min_dist; //匹配时的最小距离
    double _max_pos_change;

    //PCL
    double _grid_size;       //Voxel Grid的大小

    //3D SLAM
    double _distance_threshold; //提取平面时的距离阈值
    double _plane_percent;  //平面点的百分比
    double _min_error_plane; //归类图像点时的误差阈值
    int _max_planes; //最大平面数量

    string _loop_closure_detection; //是否使用闭环检测
    int _loopclosure_frames;   //闭环检测时随机取多少帧
    double _loop_closure_error; //闭环误差
    int _lost_frames;  //丢失判定
    string _use_odometry; //是否使用里程计
    double _error_odometry; //里程计误差
    
};

//全局变量指针
extern ParameterReader* g_pParaReader;
