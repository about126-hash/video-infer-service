#include "param.h"

int worker_count;
int port;
EngineParam EngineV8;

void readParameters(std::string config_file)
{
    //检查文件是否存在
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL)
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }
    fclose(fh);

    //创建了一个 FileStorage 对象，并以只读模式（READ）打开指定的 YAML 文件。
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    
    std::string buf;

    worker_count=fsSettings["worker_count"];
    port=fsSettings["port"];

    EngineV8.Device = fsSettings["v8_Device"];
    EngineV8.ClsNum = fsSettings["v8_ClsNum"];
    EngineV8.InputH = fsSettings["v8_InputH"];
    EngineV8.InputW = fsSettings["v8_InputW"];
    EngineV8.ConfThresh = fsSettings["v8_ConfThresh"];
    EngineV8.IouThresh = fsSettings["v8_IouThresh"];
    EngineV8.MaxInputImageSize = fsSettings["v8_MaxInputImageSize"];
    EngineV8.TensorOutputSize = fsSettings["v8_TensorOutputSize"];

    fsSettings["v8_InputTensorName"] >> buf;
    EngineV8.InputTensorName = strdup(buf.c_str());
    
    fsSettings["v8_OutputTensorName"] >> buf;
    EngineV8.OutputTensorName = strdup(buf.c_str());
    
    fsSettings["v8_engine_path"] >> buf;
    EngineV8.engine_path = strdup(buf.c_str());
    
    EngineV8.MaxNumOutputBbox = fsSettings["v8_MaxNumOutputBbox"];
    EngineV8.BatchSize = fsSettings["v8_BatchSize"];
    
    fsSettings.release();
}

