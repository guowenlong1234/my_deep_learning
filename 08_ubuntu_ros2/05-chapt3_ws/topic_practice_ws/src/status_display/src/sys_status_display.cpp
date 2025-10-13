/*
任务：通过订阅python节点发布的话题，利用自定义的消息接口拿到系统状态的相关的数据
将数据整理后通过qt进行界面展示。

*/

// 导入qt相关库文件

#include <QApplication>
#include <QLabel>
#include <QString>
#include <rclcpp/rclcpp.hpp>
#include <status_interfaces/msg/system_status.hpp>

using SystemStatus = status_interfaces::msg::SystemStatus; // 利用命名空间用SystemStatus代替status_interfaces::msg::SystemStatus

/*
builtin_interfaces/Time stamp   #记录时间戳
string host_name    #主机名称
float32 cpu_percent #cpu使用率
float32 memory_percent  #内存使用率
float32 memory_total #内存总大小
float32 memory_available    #可用内存总大小
float64 net_sent    #网络发送数据总量
float64 net_recv    #网络数据接受总量
*/
class SysStatusDisplay : public rclcpp::Node
{
private:
    rclcpp::Subscription<SystemStatus>::SharedPtr subscriber_; // 声明订阅者的共享指针
    QLabel *label_;                                            // 创建一个新的标签
public:
    SysStatusDisplay(const std::string &node_name) : Node(node_name)
    {
        label_ = new QLabel();
        subscriber_ = this->create_subscription<SystemStatus>("sys_status", 10, [&](const SystemStatus::SharedPtr msg) -> void
                                                              { label_->setText(get_qstr_from_msg(msg)); }); // 这里使用lambda表达式传入回调函数。
        label_->setText(get_qstr_from_msg(std::make_shared<SystemStatus>()));                                // 当没有消息过来时，显示空，创建了一个空的共享指针。
        label_->show();
    };
    QString get_qstr_from_msg(const SystemStatus::SharedPtr msg) // 定义一个函数，完成将msg的消息解包并将其转换成QString类型。
    {
        std::stringstream show_str;
        show_str << "================系统状态可视化工具================\n"
                 << "数 据 时 间：\t\t" << msg->stamp.sec << '\t'<<'s'<<'\n' 
                 << "主 机 名 称：\t\t" << msg->host_name << "\t\n"
                 << "CPU 使 用 率：\t\t" << msg->cpu_percent  << "\t%\n"
                 << "内 存 使 用 率：\t\t" << msg->memory_percent << "\t%\n"
                 << "内 存 总 大 小：\t\t" << msg->memory_total << "\tMb\n"
                 << "可 用 内 存：\t\t" << msg->memory_available << "\tMB\n"
                 << "数 据 发 送 量：\t\t" << msg->net_sent << "\tMB\n"
                 << "数 据 接 受 量：\t\t" << msg->net_recv << "\tMB\n"
                 << "===============================================";

            return QString::fromStdString(show_str.str());
    };
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    QApplication app(argc, argv);
    auto node = std::make_shared<SysStatusDisplay>("sys_status_display");
    // 由于rclcpp::spin(node)；与app.exec会同时阻塞代码，因此开一个新的线程来运行node节点。
    std::thread spin_thread([&]() -> void
                            {
                                rclcpp::spin(node); // 阻塞代码
                            });
    spin_thread.detach();
    app.exec();
    return 0;
};