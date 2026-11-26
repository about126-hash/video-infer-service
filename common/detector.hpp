#pragma once
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <vector>
#include "cuda_utils.h"
#include "postprocess.h"
#include "preprocess.h"
#include "config.h"


template<class T>
class Detector
{
public:
    Detector(){};

    //反序列化，构建引擎，初始化
    void Init(EngineParam input)
    {
        this->param.Device=input.Device;
        this->param.ClsNum=input.ClsNum;
        this->param.InputH=input.InputH;
        this->param.InputW=input.InputW;
        this->param.ConfThresh=input.ConfThresh;
        this->param.IouThresh=input.IouThresh;
        this->param.MaxInputImageSize=input.MaxInputImageSize;
        this->param.InputTensorName=input.InputTensorName;
        this->param.OutputTensorName=input.OutputTensorName;
        this->param.engine_path=input.engine_path;
        this->param.MaxNumOutputBbox=input.MaxNumOutputBbox;

        std::ifstream file(this->param.engine_path.c_str(), std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << this->param.engine_path << " error!" << std::endl;
            assert(false);
        }
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        char* serialized_engine = new char[size];
        assert(serialized_engine);
        file.read(serialized_engine, size);
        file.close();

        this->runtime = nvinfer1::createInferRuntime(this->gLogger);
        assert(this->runtime);
        this->engine = (this->runtime)->deserializeCudaEngine(serialized_engine, size);
        assert(this->engine);
        this->context = (this->engine)->createExecutionContext();
        assert(this->context);
        delete[] serialized_engine;

        CUDA_CHECK(cudaStreamCreate(&this->stream));
        cuda_preprocess_init(this->param.MaxInputImageSize , this->img_buffer_host, this->img_buffer_device);
        std::cout<<111<<std::endl;

        assert(this->engine->getNbBindings() == 2);
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine->getBindingIndex(this->param.InputTensorName);
        const int outputIndex = engine->getBindingIndex(this->param.OutputTensorName);

        std::cout<<222<<std::endl;
        size_t n = 1;
        nvinfer1::Dims out_dims = engine->getBindingDimensions(outputIndex);
        for (int i = 0; i < out_dims.nbDims; i++)
        {
            int d = out_dims.d[i];
            n *= d;
        }
        this->param.TensorOutputSize = n;

        std::cout<<333<<std::endl;

        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc((void**)&this->gpu_buffers[0], this->param.BatchSize * this->param.MaxInputImageSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&this->gpu_buffers[1], this->param.TensorOutputSize * sizeof(float)));
        
        this->cpu_output_buffer = new float[this->param.BatchSize * this->param.TensorOutputSize];
        
        std::cout<<"初始化完成"<<std::endl;
    }

    virtual void Preprocess()=0;


    void Infer()
    {
          // enqueueV2 需要的是"设备缓冲指针数组"(元素为 void* 设备地址)
        void* bindings[] = {gpu_buffers[0], gpu_buffers[1]};
        this->context->enqueueV2(bindings, this->stream, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(this->cpu_output_buffer, this->gpu_buffers[1], this->param.BatchSize * this->param.TensorOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    virtual void Afterprocess(T &data)=0;

    cv::Rect ToImageRect(float bbox[4])
    {
        return get_rect(this->image, bbox, this->param.InputW, this->param.InputH);
    }


    void  Run(cv::Mat input_image,T &output_data)
    {
        this->image.release();
        this->image = input_image.clone();
        this->Preprocess();
        this->Infer();
        this->Afterprocess(output_data);

    }

    virtual ~Detector()
    {
        // Release stream and buffers
        cudaStreamDestroy(stream);
        CUDA_CHECK(cudaFree(gpu_buffers[0]));
        CUDA_CHECK(cudaFree(gpu_buffers[1]));
        delete[] cpu_output_buffer;
        cuda_preprocess_destroy(this->img_buffer_host, this->img_buffer_device);
        this->context->destroy();
        this->engine->destroy();
        this->runtime->destroy();
    };

protected:
    Logger gLogger;
    // Deserialize engine
    nvinfer1::IRuntime* runtime=nullptr;
    nvinfer1::ICudaEngine* engine=nullptr;
    nvinfer1::IExecutionContext* context=nullptr;
    //cuda流
    cudaStream_t stream;    
    //缓冲区
    float* gpu_buffers[2]={nullptr,nullptr};      //gpu_buffers[0] gpu输入 gpu_buffers[1] gpu输出
    float* cpu_output_buffer = nullptr;     //cpu输出
    
    // 图像预处理缓存
    uint8_t* img_buffer_host = nullptr;
    uint8_t* img_buffer_device = nullptr;
    

    EngineParam param;
    cv::Mat image;
};

class Person_Detector:public Detector<std::vector<Detection>>
{
public:
    Person_Detector(){};

    void Preprocess()
    {
        int dst_size = this->param.InputH * this->param.InputW * 3;
        cuda_preprocess(this->image.ptr(), this->image.cols,this->image.rows, this->gpu_buffers[0], this->param.InputW , this->param.InputH, stream , this->img_buffer_host, this->img_buffer_device);
        CUDA_CHECK(cudaStreamSynchronize(stream));  
    }

    void  Afterprocess(std::vector<Detection> &output_data)
    {
        v5_nms(output_data, &this->cpu_output_buffer[0], this->param.ConfThresh, this->param.IouThresh);
    }

    ~Person_Detector(){}
};

class v8_Detector:public Detector<std::vector<Detection>>
{

public:
    v8_Detector(){};

    void Preprocess()
    {
        int dst_size = this->param.InputH * this->param.InputW * 3;
        cuda_preprocess(this->image.ptr(), this->image.cols,this->image.rows, this->gpu_buffers[0], this->param.InputW , this->param.InputH, stream , this->img_buffer_host, this->img_buffer_device);
        CUDA_CHECK(cudaStreamSynchronize(stream));  
    }

    void Afterprocess(std::vector<Detection> &output_data)
    {
        v8_nms(output_data, &this->cpu_output_buffer[0], this->param.ClsNum , this->param.MaxNumOutputBbox , this->param.ConfThresh, this->param.IouThresh);
    }

    ~v8_Detector(){}
};


class Path_Detector:public Detector<cv::Mat>
{

public:
    Path_Detector(){};

    void Preprocess()
    {
        // float * data=new float[this->param.BatchSize * 3 * this->param.InputH * this->param.InputW];
        
        std::unique_ptr<float[]> data = std::make_unique<float[]>(this->param.BatchSize * 3 * this->param.InputH * this->param.InputW);

        for (int i = 0; i < this->param.InputH * this->param.InputW; i++) 
        {
            data[i] = (this->image.at<cv::Vec3b>(i)[2]) / 255.0;
            data[i + this->param.InputH * this->param.InputW] = (this->image.at<cv::Vec3b>(i)[1]) / 255.0;
            data[i + 2 * this->param.InputH * this->param.InputW] = (this->image.at<cv::Vec3b>(i)[0]) / 255.0;
        }
        float * data_ptr = data.get();
        CHECK(cudaMemcpyAsync(this->gpu_buffers[0], data_ptr, this->param.BatchSize * 3 * this->param.InputH * this->param.InputW* sizeof(float), cudaMemcpyHostToDevice, this->stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));  
        // delete []data;
    }

    void  Afterprocess(cv::Mat &output_data)
    {
        cv::Mat result;
        result=cv::Mat::zeros(this->param.InputH , this->param.InputW ,CV_8UC1);

        for (int i = 0; i < this->param.InputH * this->param.InputW; i++) {
            float fmax = this->cpu_output_buffer[i];
            int index = 0;
            for (int j = 1; j < 4; j++) {
                if (this->cpu_output_buffer[i + j * this->param.InputH * this->param.InputW] > fmax) {
                    index = j;
                    fmax = this->cpu_output_buffer[i + j * this->param.InputH * this->param.InputW];
                }
            }
            result.at<uchar>(i) =index;
        }
        output_data = result.clone();

        
    }
    ~Path_Detector(){}
};






