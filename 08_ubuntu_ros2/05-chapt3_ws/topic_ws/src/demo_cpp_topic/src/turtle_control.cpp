/*
核心需求：
实现闭环控制小海龟到达指定地点
速度策略：以当前点和目标点之间的距离作为速度。
角速度策略：以当前朝向和目标点方向的角度值作为角速度。
通过订阅话题获取当前小海龟的速度与朝向
在经过计算后给出角速度以及线速度发出话题控制小海龟运动。
小海龟节点通过/turtle1/pose [turtlesim/msg/Pose]话题来发布自己的当前位置信息。
*/
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <chrono>
#include "turtlesim/msg/pose.hpp"

class TurtleControlNode : public rclcpp::Node
{
private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_; // 声明发布者的智能指针
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr subscriber_;  // 声明订阅者的共享指针
    double target_x_{1.0};
    double target_y_{1.0};
    double k_{1.0};         // 比例系数
    double max_speed_{1.0}; // 限制最大速度

public:
    explicit TurtleControlNode(const std::string &node_name) : Node(node_name)
    {
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10); // 实例化一个发布者
        subscriber_ = this->create_subscription<turtlesim::msg::Pose>("/turtle1/pose", 10, std::bind(&TurtleControlNode::on_pose_received_, this, std::placeholders::_1));
    }
    void on_pose_received_(const turtlesim::msg::Pose::SharedPtr pose) // 参数：受到数据的共享指针
    {
        auto msg = geometry_msgs::msg::Twist();
        // 获取到当前位置
        auto current_x = pose->x;
        auto current_y = pose->y;
        RCLCPP_INFO(get_logger(), "当前:x=%f, y=%f", current_x, current_y);

        // 2.计算当前海龟位置与目标位置之间的距离和角度
        auto distance = std::sqrt((target_x_ - current_x) * (target_x_ - current_x) + (target_y_ - current_y) * (target_y_ - current_y));
        auto angle = std::atan2((target_y_ - current_y),(target_x_ - current_x)) - pose->theta;

        //3.控制策略
        if(distance>0.1){
            if(fabs(angle)>0.2){
                msg.angular.z = fabs(angle);
            }else{
                msg.linear.x = k_*distance;
            }
        }
        //4.限制线速度最大值
        if(msg.linear.x>max_speed_){
            msg.linear.x = max_speed_;
        }
        //5.发布控制话题
        publisher_->publish(msg);

    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TurtleControlNode>("turtle_control");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}