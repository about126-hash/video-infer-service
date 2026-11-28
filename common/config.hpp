#pragma once
#include <iostream>

struct EngineParam
{
public:
    EngineParam(){};
    EngineParam& operator=(const EngineParam&) = delete;


    int         Device; //gpu  0 cpu 1
    int         ClsNum; 
    int         InputH; //输入图像尺寸           
    int         InputW;
    float       ConfThresh;   //置信度
    float       IouThresh;    //用于目标检测模型，实例分割不使用
    int         MaxInputImageSize;//->InputH*this->InputW*2*2;  //图像最大尺寸，用于cuda初始化,预留2×2倍
    int         TensorOutputSize; //输出张量大小
    char*       InputTensorName;
    char*       OutputTensorName;
    std::string engine_path;   //引擎路径
    
    int MaxNumOutputBbox=1000;   //目标检测模型所能输出的最大框数量，影响cpu_output，设置1000，预留空间，防止崩溃，密集目标，可在拉高此值
    int BatchSize=1;
};

