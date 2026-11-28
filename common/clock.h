#include <iostream>
#include <chrono>

template<class T>
class clock
{
private:
    std::chrono::high_resolution_clock::time_point start_time;  //不强制指定milliseconds或者其他时间单位
    clock(const clock&) = delete;

    clock& operator=(const clock&) = delete;
public:
    clock()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }
    float GetCostTime()
    {   
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<T>(end_time - start_time).count();
    }

    void Reset()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    //后续可以考虑通过RAII范式，添加析构函数
    ~clock() = default;
};