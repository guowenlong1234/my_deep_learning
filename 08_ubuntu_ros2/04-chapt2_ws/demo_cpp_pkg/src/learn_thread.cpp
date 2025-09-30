#include <iostream>
#include <thread>
#include <chrono>               //时间相关的头文件
#include <functional>           //函数包装器
#include "cpp-httplib/httplib.h" //下载相关的头文件

class Download
{
private:
    /* data */
public:
    void download(const std::string &host, const std::string &path, const std::function<void(const std::string &, const std::string &)> &callback)
    {
        std::cout << "线程" << std::this_thread::get_id() << std::endl;
        httplib::Client client(host);
        auto response = client.Get(path);
        if (response && response->status == 200)
        {
            callback(path, response->body);
        }
    };

    void start_download(const std::string &host, const std::string &path, const std::function<void(const std::string &, const std::string &)> &callback) {
        auto download_fun = std::bind(&Download::download,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3);
        std::thread thread(download_fun,host,path,callback);//创建线程，目标函数是包装过的download_fun，采用的成员函数包装方式。用的bind函数。在cpp中，线程创建后会立即运行，并且会阻塞当前线程。
        thread.detach();//将当前线程从线程中剥离出来。防止阻塞当前线程。

    };
};

int main()
{
    auto d = Download();
    auto word_count = [](const std::string &path, const std::string &result) -> void
    {
        std::cout << "下载完成" << path << ";" << result.length() << "->" << result.substr(0, 9) << std::endl;
    };
    d.start_download("http://0.0.0.0:8000", "/novel1.txt", word_count);
    d.start_download("http://0.0.0.0:8000", "/novel2.txt", word_count);
    d.start_download("http://0.0.0.0:8000", "/novel3.txt", word_count);

    std::this_thread::sleep_for(std::chrono::milliseconds(1000 * 10)); // 让主进程休眠1000*10毫秒，防止进程被杀死。
    return 0;
}