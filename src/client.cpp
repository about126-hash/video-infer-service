#include "httplib.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
// 有界阻塞队列 
template <class T>
class BlockQueue
{
public:
    explicit BlockQueue(int timeout_ms , size_t cap = 8) : timeout_ms_(timeout_ms), cap_(cap), done_(false) {}

    // 满则阻塞
    bool Push(T&& item)
    {
        std::unique_lock<std::mutex> lock(mtx_);

        if(!notFull_.wait_for(lock, std::chrono::milliseconds(timeout_ms_), [this] { return q_.size() < cap_ || done_; }))
        {
            return false; 
        }

        if (done_) return false; 

        q_.push(std::move(item));
        notEmpty_.notify_one();
        return true;
    }

    bool Pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        if(!notEmpty_.wait_for(lock, std::chrono::milliseconds(timeout_ms_), [this] { return !q_.empty() || done_; }))
        {
            return false; 
        }

        if (q_.empty()) return false; 

        item = std::move(q_.front());
        q_.pop();
        notFull_.notify_one();
        return true;
    }

    // 唤醒所有阻塞中的 Push/Pop
    void Stop()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            done_ = true;
        }
        notEmpty_.notify_all();
        notFull_.notify_all();
    }

private:
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable notFull_, notEmpty_;
    int timeout_ms_;
    size_t cap_;
    bool done_;
};



int main(int argc, char** argv)
{
    // 用法: ./client [服务端IP=127.0.0.1] [端口=8080] [摄像头索引=0] [JPEG质量=95]
    if (argc > 1 && std::string(argv[1]) == "-h")
    {
        std::cout << "用法: " << argv[0] << " [服务端IP=127.0.0.1] [端口=8080] [摄像头索引=0] [JPEG质量=95]\n"
                  << "示例: " << argv[0] << " 192.168.1.100 8080 0 85\n";
        return 0;
    }

    const std::string host = (argc > 1) ? argv[1] : "127.0.0.1";
    const int         port = (argc > 2) ? atoi(argv[2]) : 8080;
    const int         cam_idx = (argc > 3) ? atoi(argv[3]) : 0;
    const int         jpeg_quality = (argc > 4) ? atoi(argv[4]) : 95;

    cv::VideoCapture cap(cam_idx, cv::CAP_V4L2);

    if (!cap.isOpened())
    {
        std::cerr << "摄像头打开失败（index=" << cam_idx << "）！\n";
        return 1;
    }

    //指定分辨率
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280/2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 960/2);
    cap.set(cv::CAP_PROP_FPS,          25);
    std::cout << "摄像头: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " @"
              << cap.get(cv::CAP_PROP_FPS) << "fps，服务端 "
              << host << ":" << port << "，JPEG质量 " << jpeg_quality << "\n";

    cv::Mat img;

    BlockQueue<cv::Mat> img_queue(1,1);
    std::atomic<bool>running{true};

    std::thread clinet_thread([&img_queue,&running,&host,port,jpeg_quality]{
        httplib::Client cli(host, port);
        cli.set_connection_timeout(5);
        cli.set_read_timeout(10);
        cli.set_keep_alive(true);  //开启长连接，库自动处理Connection:keep-alive头

        std::vector<uint8_t> buf;
        std::vector<int> param;
        param.push_back(cv::IMWRITE_JPEG_QUALITY);
        param.push_back(jpeg_quality);

        cv::Mat img;

        int frame_cnt =0;
        float frame_time_ms = 0.0f;
        while (running.load())
        {
            buf.clear();
            auto t0 = std::chrono::steady_clock::now();

            if(!img_queue.Pop(img))
            {
                //为了防止此处长期轮空，增加开销，添加延时
                //BlockQueue Pop中，已进行阻塞，此处在添加，则为A+B,保留一个，根据程序实际需求进行调整
                //此处为了方便，将延迟注释掉
                // std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            cv::imencode(".jpg", img, buf, param);

            std::string body(buf.begin(), buf.end());

            auto res = cli.Post("/infer", body, "image/jpeg");

            if (!res)
            {
                std::cerr << "网络错误：" << httplib::to_string(res.error()) << "\n";
                // 长连接断了之后，下一轮Post库内部会自动尝试重连，这里不直接return
                continue;
            }

            std::cout << "status=" << res->status << "\n";
            // 查看服务端返回的Connection头，验证keep-alive是否协商成功
            if (res->has_header("Connection"))
            {
                std::cout << "resp Connection: " << res->get_header_value("Connection") << "\n";
            }
            std::cout << "response:\n" << res->body << "\n\n";

            auto t1 = std::chrono::steady_clock::now();
            double cost_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            frame_time_ms += cost_ms;

            if(++frame_cnt % 100 == 0)
            {
                std::cout << "client avg cost_ms=" << frame_time_ms / 100 << "\n";
                frame_time_ms = 0.0f;
                frame_cnt = 0;
            }
            
            //此处不作为显示线程，所以不需要添加cv::waitKey(),如果需要添加，需要确保cv::waitKey()在每个循环中都会被进行调用
            
        }
        img_queue.Stop();
    });

    int frame_all =0;   //统计全部图像帧数
    int frame_cnt =0;   //统计成功Push的图像帧数
    float frame_time_ms = 0.0f;
    while (true)
    {
        auto t0 = std::chrono::steady_clock::now();

        cv::Mat img;
        cap >> img;
        if (img.empty())
        {
            std::cerr << "读取摄像头帧失败！\n";
            continue;
        }

        //成功推送图像帧后的业务逻辑
        if(img_queue.Push(std::move(img)))
        {
            frame_cnt++;
        }

        auto t1 = std::chrono::steady_clock::now();
        double cost_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        frame_time_ms += cost_ms;

        if(++frame_all % 100 == 0)
        {
            std::cout << "捕获100帧图像 cost_ms=" << frame_time_ms / 100 << "\n";
            std::cout << "成功Push的图像帧数 cnt=" << frame_cnt << "\n";
            frame_time_ms = 0.0f;
            frame_all = 0;
            frame_cnt = 0;
        }

    }

    img_queue.Stop();
    running.store(false);
    clinet_thread.join();

    return 0;
}
