#pragma once
#include <iostream>
#include "types.h"


struct EngineParam
{
public:
    EngineParam(){};
    EngineParam(
        int   in_Device, //gpu  0 cpu 1
        int   in_ClsNum, 
        int   in_InputH, //输入图像尺寸           
        int   in_InputW,
        float in_ConfThresh,   //置信度
        float in_IouThresh,    //用于目标检测模型，实例分割不使用
        char* in_InputTensorName,
        char* in_OutputTensorName,
        std::string in_engine_path,   //引擎路径
        int   in_MaxNumOutputBbox,
        int   in_BatchSize)
    {
        this->Device = in_Device;
        this->ClsNum = in_ClsNum;
        this->InputH = in_InputH;
        this->InputW = in_InputW;
        this->ConfThresh = in_ConfThresh;
        this->IouThresh = in_IouThresh;
        this->MaxInputImageSize = in_InputH*in_InputH*3;//->InputH*this->InputW*2*2;  //图像最大尺寸，用于cuda初始化,预留2×2倍
        this->InputTensorName = in_InputTensorName;
        this->OutputTensorName = in_OutputTensorName;
        this->engine_path = in_engine_path;
        this->MaxNumOutputBbox = in_MaxNumOutputBbox;
        this->BatchSize = in_BatchSize;


    }
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


static  EngineParam EngineV8(
 0, //in_Device
 80, //in_ClsNum
 640,   //in_InputH
 640,   //in_InputW
 0.3,   //in_ConfThresh
 0.3,   //in_IouThresh
 "images",    //in_InputTensorName
 "output0",    //in_OutputTensorName
 "/home/lhl/cpp/Video_infer_service/models/yolov8n_fp32.engine",    //in_engine_path
 8400,  //in_MaxNumOutputBbox
 1  //in_BatchSize
);



