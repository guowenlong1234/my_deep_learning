#include "rclcpp/rclcpp.hpp"

class PersonNode : public rclcpp::Node // c++中的继承写法 class与public是关键词
{
private: // 私有成员声明
    /* data */
    std::string name_; // 私有成员名字
    int age_;          // 私有成员年龄

public: // 可以公开调用的
    /*
    用类的名字定义构造函数，中间的参数与python中__init__相同，传入实例化类需要的参数
    :Node与python中supur函数相同，调用父类的构造函数，需要传入对应的参数
    先传递近来参数，再将参数传递给父类。
    const std::string & name, const int & age
    const 类型 & 参数名，const 类型 & 参数名
    这里的传递方式是引用传递方式，因此采用了&符号，引用传递相较于值传递来说，减少了对数值不必要的复制移动操作，提高了代码的性能
    采用const方法，使变量变为只读变量，提高了代码安全性
    */
    PersonNode(const std::string & node_name, const std::string &name, const int &age) : Node("node_name")

    {
        this->name_ = name; // 相当于self.name = name
        this->age_ = age;   // self.age = age
    };

    /*
    c++中传递参数本质上是传递的指针，因此在函数内部可以对参数进行修改，为了避免这种情况的发生，采用const，使在函数内部无法赋值

    */
    void eat(const std::string &food_name) // 声明一个eat方法，传入参数food_name
    {
        RCLCPP_INFO(this -> get_logger(), "我是%s,%d岁，爱吃%s", this->name_.c_str(), this->age_, food_name.c_str());
    };
};

int main(int argc,char** argv)
{
    rclcpp::init(argc , argv);
    auto node = std::make_shared<PersonNode>("person_node", "李斯", 18);    //实例化类
    RCLCPP_INFO(node->get_logger(),"this is c++ node");

    node->eat("蛋炒饭");        //node是指针，需要使用->来指向其中的成员方法
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}