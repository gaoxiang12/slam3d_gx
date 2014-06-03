#include "ParameterReader.h"
#include <fstream>
#include "yaml-cpp/yaml.h"
#include "const.h"

using namespace std;

ParameterReader* g_pParaReader;
double camera_fx, camera_fy,  camera_cx, camera_cy, camera_factor;

ParameterReader::ParameterReader(const string& para_file )
{
    cout<<"init parameterReader, file addr = "<< para_file<<endl;

    ifstream fin(para_file.c_str());

    YAML::Parser parser(fin);
    YAML::Node config;
    try {
        parser.GetNextDocument(config);
    } catch (YAML::ParserException& e)
    {
        cerr<<e.what()<<"\n";
        return ;
    }

    //直接从config中读信息即可
    config[ "data_source" ] >> _data_source;
    config[ "detector_name" ] >> _detector_name;
    config[ "descriptor_name" ] >> _descriptor_name;
    config[ "start_index" ] >> _start_index;
    config[ "end_index" ] >> _end_index;
    config[ "step_time" ] >> _step_time;
    config[ "optimize_step" ] >> _optimize_step;
    config[ "robust_kernel" ] >> _robust_kernel;
    if (_end_index < _start_index)
    {
        cerr<<"end index should be larger than start index."<<endl;
        return;
    }

    config[ "max_pos_change" ] >> _max_pos_change;
    config[ "step_time_keyframe" ] >> _step_time_keyframe;
    config[ "error_threshold" ] >> _error_threshold;
    config[ "grid_leaf" ] >> _grid_size;

    config[ "distance_threshold" ] >> _distance_threshold;
    config[ "plane_percent" ] >> _plane_percent;
    config[ "min_error_plane" ] >> _min_error_plane;
    config["max_planes"] >> _max_planes;
    config[ "loop_closure_detection" ] >> _loop_closure_detection;
    config[ "loopclosure_frames"] >> _loopclosure_frames;
    config[ "loop_closure_error"] >> _loop_closure_error;

    config[ "camera_fx" ] >> camera_fx;
    config[ "camera_fy" ] >> camera_fy;
    config[ "camera_cx" ] >> camera_cx;
    config[ "camera_cy" ] >> camera_cy;
    config[ "camera_factor" ] >> camera_factor;

    config[ "lost_frames" ] >> _lost_frames;
}

string ParameterReader::GetPara( const string& para_name )
{
    if (para_name == string("detector_name"))
        return _detector_name;
    if (para_name == string("descriptor_name"))
        return _descriptor_name;
    if (para_name == string("data_source"))
        return _data_source;
    if (para_name == string("step_time"))
        return num2string(_step_time);
    if (para_name == string("save_if_seen"))
        return num2string(_save_if_seen);
    if (para_name == string("del_not_seen"))
        return num2string( _del_not_seen );
    if (para_name == string("optimize_step"))
        return num2string(_optimize_step);
    if (para_name == string("robust_kernel"))
        return _robust_kernel;
    if (para_name == string("set_max_depth"))
        return _set_max_depth;
    if (para_name == string("match_min_dist"))
        return num2string(_match_min_dist);
    if (para_name == string("max_landmark_per_loop"))
        return num2string(_max_landmark_per_loop);
    if (para_name == string("max_pos_change"))
        return num2string(_max_pos_change);
    if (para_name == string("grayscale"))
        return _grayscale;
    if (para_name == string("start_index"))
        return num2string(_start_index);
    if (para_name == string("end_index"))
        return num2string(_end_index);
    if (para_name == string("step_time_keyframe"))
        return num2string(_step_time_keyframe);
    if (para_name == string("online_pcl"))
        return _online_pcl;
    if (para_name == string("fabmap"))
        return _fabmap;
    if (para_name == string("error_threshold"))
        return num2string(_error_threshold);
    if (para_name == string("grid_leaf"))
        return num2string(_grid_size);
    if (para_name == string("distance_threshold") )
        return num2string(_distance_threshold);
    if (para_name == string("plane_percent"))
        return num2string(_plane_percent);
    if (para_name == string("min_error_plane") )
        return num2string(_min_error_plane );
    if (para_name == string("max_planes"))
        return num2string( _max_planes );
    if (para_name == string("loopclosure_frames"))
        return num2string( _loopclosure_frames );
    if (para_name == string("loop_closure_detection"))
        return _loop_closure_detection;
    if (para_name == string("loop_closure_error"))
        return num2string(_loop_closure_error);
    if (para_name == string("lost_frames"))
        return num2string(_lost_frames);
    cerr<<"Unknown parameter: "<<para_name<<endl;
    return string("unknown_para_name");
}
