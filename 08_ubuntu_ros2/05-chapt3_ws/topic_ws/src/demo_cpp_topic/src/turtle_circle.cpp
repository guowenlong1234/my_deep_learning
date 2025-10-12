/*
任务目标：
发布话题/turtle1/cmd_vel来控制小海龟画圆
*/

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <chrono>

using namespace std::chrono_literals;

class TurtlrCircleNode : public rclcpp::Node
{
private:
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_; // 声明一个发布者的智能指针
public:
    explicit TurtlrCircleNode(const std::string &node_name) : Node(node_name)
    {
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);       // 实例化一个发布者的智能指针。
        timer_ = this->create_wall_timer(1000ms, std::bind(&TurtlrCircleNode::timer_callback, this)); // 成员函数调用函数包装器的用法。
    }

    void timer_callback() // 成员函数作为回调函数，每次到了时间就执行一下这个回调函数。
    {
        auto msg = geometry_msgs::msg::Twist(); //创建一个消息对象，采用自动推导类型。
        msg.linear.x = 1.0;
        msg.angular.z = 0.5;
        publisher_->publish(msg);    //智能指针指向下的成员函数或方法要用->运算符，而调用对象本身的属性或方法应该采用.运算符。

    }
};

int main(int argc,char* argv[])
{
    rclcpp::init(argc,argv);
    auto node = std::make_shared<TurtlrCircleNode>("turtle_circle");
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}