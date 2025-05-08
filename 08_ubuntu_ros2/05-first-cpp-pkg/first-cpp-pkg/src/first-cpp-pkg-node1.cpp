
// #include "iostream"
#include "rclcpp/rclcpp.hpp"
//argc和argv用来控制命令行启动程序时候用户输入的命令，argc用于存储参数的数量，
//argv是一个二维数组用于存储每一个参数的值，参数通过空格进行分割

int main(int argc,char** argv)
{   
    rclcpp::init(argc,argv);    //初始化ros2节点
    auto node = std::make_shared<rclcpp::Node>("cpp_node");  //实例化节点，node是一个智能指针
    RCLCPP_INFO(node->get_logger(),"cpp node is running");  //打印一条日志
    rclcpp::spin(node);         //启动节点
    rclcpp::shutdown();
    // std::cout<<"参数数量="<<argc<<std::endl;
    // std::cout<<"程序名字="<<argv[0]<<std::endl;
    // std::string arg1 = argv[1];
    

    // if(arg1=="--help")
    // {
    //     std::cout<<"这里是程序帮助"<<std::endl;
    // }
    return 0;
}