#include "httplib.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <vector>
#include "detector.hpp"
#include "config.hpp"
#include <chrono>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>
#include <nvtx3/nvToolsExt.h>
#include <param.h>
#include <csignal>

class NvtxRange
{
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange&) = delete;            // 禁止拷贝
    NvtxRange& operator=(const NvtxRange&) = delete; // 否则两份对象析构两次 pop
};

//用DeadlineExceeded替代std::runtime_error("deadline exceeded")，这种形式更严格，类型明确吗
struct DeadlineExceeded : std::runtime_error
{
    DeadlineExceeded() : std::runtime_error("deadline exceeded") {}
};



const std::vector<std::string> coco_class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct Box
{
    int x1, y1, x2, y2;
    float score;
    int cls;
};



//有界阻塞队列 
template <class T>
class BlockQueue
{
public:
    explicit BlockQueue(int timeout_ms , size_t cap = 8) : timeout_ms_(timeout_ms), cap_(cap), done_(false) {}

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

        if (q_.empty()) return false; // 

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

struct Task
{
    cv::Mat image;
    std::chrono::steady_clock::time_point deadline;
    std::promise<std::vector<Box>> promise;

    std::future<std::vector<Box>> GetFuture() { return promise.get_future(); }
};


httplib::Server* g_svr = nullptr;


void signalHandler(int signum) {
    //终止掉原有的svr阻塞，.linsen()结束后，会继续执行资源释放类操作，然后结束main()
    if(g_svr) g_svr->stop();
}

int main()
{
    //signal csignal库中用于处理信号的核心函数，可为特别信号指定处理函数，如SIGINT(程序中断信号(ctrl+c)),SIGTERM(程序终止信号(kill))等
    //param1 信号类型  param2 信号处理函数指针
    //捕获ctrl+c信号，覆盖原有的程序终止行为
    std::signal(SIGINT, signalHandler);


    readParameters("/home/lhl/cpp/Video_infer_service/config/config.yaml");
 
    if (worker_count <= 0)
    {
        std::cout<<"线程数配置错误，程序退出"<<std::endl;
        return 0;
    }
    if (port <= 0 || port > 65535) 
    {
        std::cout<<"端口配置错误，程序退出"<<std::endl;
        return 0;
    };

    BlockQueue<Task> queue(10,200);
    BlockQueue<cv::Mat> show_queue(5,20);

    std::atomic<bool>running{true};

    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    for(int i = 0; i < worker_count; i++)
    {
        workers.emplace_back([i,&queue,&running,&EngineV8]
        {
            int worker_id = i;
            std::printf("worker 线程 id: %d\n", worker_id);

            std::unique_ptr<Detector<std::vector<Detection>>> det(new v8_Detector());
            std::vector<Detection> raw;

            det->Init(EngineV8);

            cv::Mat black_buf(cv::Size(640,640), CV_8UC3, cv::Scalar(0,0,0));
            det->Run(black_buf, raw);

            Task task;
            while (running.load())
            {
                raw.clear();

                if(!queue.Pop(task))
                {
                    continue;
                }

                // 防止积压，任务超时
                if (std::chrono::steady_clock::now() > task.deadline)
                {
                    task.promise.set_exception(
                        std::make_exception_ptr(DeadlineExceeded{}));
                    continue;
                }

                try
                {
                    det->Run(task.image, raw);

                    std::vector<Box> boxes;
                    for (auto& d : raw)
                    {
                        cv::Rect r = det->ToImageRect(d.bbox);
                        boxes.push_back({r.x, r.y, r.x + r.width, r.y + r.height,
                                        d.conf, static_cast<int>(d.class_id)});
                    }
                    task.promise.set_value(std::move(boxes));
                }
                catch (...)
                {
                    // 不让异常把 promise 悬空，否则 main 的 future.get() 会抛 broken_promise
                    task.promise.set_exception(std::current_exception());
                }
            }

            std::printf("worker%d 退出\n", worker_id);

        });
    }


    std::thread show_thread([&show_queue,&running]
    {
        cv::Mat image;
        while(running.load())
        {
            if(show_queue.Pop(image))
            {
                cv::imshow("show", image);            
            }
            cv::waitKey(10);
        }

    });


    httplib::Server svr;
    g_svr = &svr;

    svr.Post("/infer", [&queue,&show_queue](const httplib::Request& req, httplib::Response& res)
    {
        std::cout << "===== 收到新请求 =====" << std::endl;

        auto it = req.headers.find("Content-Type");
        std::string ctype = "image/png";
        if (it != req.headers.end())
        {
            ctype = it->second;
            std::cout << "Content-Type: " << ctype << std::endl;
        }
        else
        {
            std::cout << "Content-Type: (无)" << std::endl;
        }

        // req.body.size() 请求体的字节长度
        std::cout << "body 字节长度: " << req.body.size() << std::endl;

        if (req.body.empty())
        {
            res.status = 400;
            res.set_content(R"({"code":-1,"msg":"body为空，未上传图片"})", "application/json");
            return;
        }

        auto t_start = std::chrono::steady_clock::now();

        cv::Mat buffer(1, req.body.size(), CV_8UC1, (void*)req.body.data());

        cv::Mat image;
        {
            NvtxRange r("imdecode");
            image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        }

        if(image.empty())
        {
             res.status = 400;
            res.set_content(R"({"code":-1,"msg":"图片解码失败，不是有效图片"})", "application/json");
            return;
        }

        Task task;
        task.image = image;
        
        //任务超时时间>200ms,则为超时
        task.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        auto fut = task.GetFuture();

        std::vector<Box> boxes;
        {
            NvtxRange r("push+wait");

            if (!queue.Push(std::move(task)))
            {
                res.status = 503;
                res.set_content(R"({"code":-2,"msg":"queue closed"})", "application/json");
                return;
            }

            //wait_for()判断超时,get()取结果
            if (fut.wait_for(std::chrono::seconds(2)) == std::future_status::timeout)
            {
                res.status = 504;
                res.set_content(R"({"code":-504,"msg":"timeout, please try again later"})", "application/json");
                return;
            }

            try
            {
                boxes = fut.get();   //wait_for 已确认 ready，这里不阻塞；worker 内异常在此重新抛出
            }
            catch (const DeadlineExceeded&)
            {
                res.status = 504;
                res.set_content(R"({"code":-504,"msg":"request expired in queue"})", "application/json");
                return;
            }
            catch (const std::exception& e)
            {
                res.status = 500;
                res.set_content(nlohmann::json{{"code", -500}, {"msg", e.what()}}.dump(), "application/json");
                return;
            }
        }


        //构建json
        nlohmann::json resp;
        nlohmann::json boxes_arr = nlohmann::json::array();

        for (auto& d : boxes)
        {
            nlohmann::json box;
            box["x1"] = d.x1;
            box["y1"] = d.y1;
            box["x2"] = d.x2;
            box["y2"] = d.y2;
            box["score"] = d.score;
            box["cls"] = d.cls;
            boxes_arr.push_back(box);

            cv::Rect rect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
            cv::putText(image, coco_class_names[d.cls], cv::Point(rect.x, rect.y - 1),
            cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 0xFF), 2);
        }

        resp["boxes"] = boxes_arr;

        // // 一个请求，一个线程，因此此处调用imshow,waitKey会跨线程调用，不安全，需要进行同步
        // cv::imshow("recv_img", image);
        // cv::waitKey(1);

        show_queue.Push(std::move(image));


        auto t_end = std::chrono::steady_clock::now();
        double cost_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        resp["ms"] = cost_ms;


        ctype = "application/json";
        res.set_content(resp.dump(), ctype);
    });

    svr.Get("/", [](const httplib::Request& req, httplib::Response& res)
    {
        res.set_content("service ready, use POST /infer to upload image", "text/plain");
    });

    svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep)
    {
        std::string msg = "unknown exception";
        try
        {
            if (ep) std::rethrow_exception(ep);
        }
        catch (const std::exception& e) { msg = e.what(); }
        catch (...) {}
        std::cerr << "请求处理异常: " << msg << std::endl;
        res.status = 500;
        res.set_content(nlohmann::json{{"code", -500}, {"msg", msg}}.dump(), "application/json");
    });


    
    svr.set_tcp_nodelay(true);
    svr.set_keep_alive_timeout(8);  //单位：s

    std::cout << "服务启动: 0.0.0.0:" << port << "（worker数=" << worker_count << "）" << std::endl;
    svr.listen("0.0.0.0", port);

    //在程序内部手动模拟信号
    // raise(SIGINT);


    queue.Stop();
    show_queue.Stop();
    running.store(false);
    for (auto& w : workers)
    {
        if (w.joinable()) w.join();
    }
    show_thread.join();
    std::cout<<"程序退出"<<std::endl;

    // SIGINT：表示程序接收到中断信号，通常由用户按下Ctrl+C触发。
    // 此信号用于通知程序用户希望中断当前操作。例如，在命令行程序中，用户按下Ctrl+C时，程序会收到SIGINT信号，可以捕获此信号以优雅地关闭程序。


    return 0;
}
