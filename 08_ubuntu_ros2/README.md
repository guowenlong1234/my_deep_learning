# 1.ubuntu基本操作
记录一下各种命令，方便以后重装
## 常用指令
修改环境变量指令
```
export PATH_NAME=PATH_VALUE
export RCUTILS_CONSOLE_OUTPUT_FORMAT=[{function_name}:{line_number}]:{message}
```
打印全部环境变量
```
printenv | grep PATH_NAME
```

## 常用环境变量
控制ros2中日志打印格式，

- RCUTILS_CONSOLE_OUTPUT_FORMAT=[{function_name}:{line_number}]:{message}
## 刚更新完成之后的操作

- 设置时钟

由于win与ubuntu时钟设置不同，修改ubuntu时钟方式为读取本地主板时间。
[设置系统时间看这里](https://www.bilibili.com/video/BV1554y1n7zv)

```
sudo apt install ntpdate
```
```
sudo ntpdate time.windows.com
```
```
sudo hwclock --localtime --systohc
```

- 修改启动引导项（解决开机黑屏无显卡驱动问题以及默认启动系统）

linux启动时，出现显卡驱动无法加载问题，多见于新显卡。解决方法禁用linux自带的第三方显卡驱动。第一次启动时在ubuntu启动项按e设置启动参数，在linux~~~~quiet splash ---这里改为quiet splash nomodesate在进行启动。进入之后将启动配置文件修改，配置文件在/etc/default/grub.#将GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"修改为
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nomodesate"。[开机启动项设置看这里](https://www.bilibili.com/video/BV1554y1n7zv)
```
sudo gedit /etc/default/grub
```
```
sudo update-grub
```
修改引导默认启动系统选项，记住自己的windows系统的序号，将GRUB_DEFAULT=0 改为 GRUB_DEFAULT=3,每次数10s后默认启动windows

# 2.关于v2ray的使用
下载链接：[v2ray客户端下载](https://github.com/selierlin/Share-SSR-V2ray/wiki/Linux%E4%BD%BF%E7%94%A8SS)

下载完成后在终端启动代理服务。注：其他软件想使用代理服务需要各自设置代理规则。默认代理端口为10808,可在客户端>设置>参数设置中修改

开启终端代理代码，可以写入./bashrc文件
```
export http_proxy=http://127.0.0.1:1080
```
```
export https_proxy=http://127.0.0.1:1080
```
```
export socks5_proxy=sockes5://127.0.0.1:1080
```
# 关于C++
c++基本类的用法参见[这里](./07-cpp-learning/cpp-learn/src/cpp-learn.cpp)
# 关于ros2 pakge构建方法
## python版本
- 01.构建功能包指令
```
ros2 pkg create --build-type ament_python --license Apache-2.0 <pkgname>
```
- 02.编写功能包代码放在pkg_name路径下
- 03.在setup.py文件中声明功能包的路径以及可执行文件的名称
构建好功能包之后，在<pkg_name>路径下新建.py文件，然后在setup.py文件中声明节点。声明节点的代码
```
    entry_points={
        'console_scripts': [
            "first_python_pkg = first_python_pkg.first_python_pkg_node1:main"       #这一行是新加的
            "name.exe         = pkg_name.node_name:function_name"    #指明生成的可执行文件名称，以及对应的函数位置。
        ],
    },
```
- 04.在package.xml(功能包清单文件)文件中添加依赖声明
```
<depend>rclpy</depend>
```
- 05.命令行中使用colcon命令构建功能包
```
colcon build
```
注：colon会在当前文件路径下扫描所有的功能包并进行构建，在这条命令的路径下生成三个子文件夹。其中build文件夹为构建过程中产生的中间文件。install文件夹为构建结果文件夹，其下有一个功能包文件夹
- 06.添加环境变量使得脚本能够找到度英功能包的路径,在install路径下自带有setup.bash文件可以修改环境变量。
```
source setup.bash
```
## c++版本
- 01.构建功能包指令
```
ros2 pkg create --build-type ament_cmake --license Apache-2.0 <pkgname>
```
- 02.编写节点代码
在src路径下添加新的node代码文件。
- 03.CMakeLists.txt文件中指定代码,指定内容与构建节点时所用的基本相同，
```
add_executable(ros2_cpp_node src/ros2-first-cpp-node.cpp)       #(生成的可执行文件名称 等待编译的文件相对路径)

find_package(rclcpp REQUIRED)       #直接查找到rclcpp头文件和库文件所在的路径，REQUIRED表示这个文件是必须的,会将他暂时存在类似于环境变量中头文件rclcpp_INCLUDE_DIRS等待调用

message(STATUS ${rclcpp_INCLUDE_DIRS})  #头文件存储路径
message(STATUS ${rclcpp_LIBRARIES})  #库文件存储路径librclcpp.so 动态链接库

target_include_directories(ros2_cpp_node PUBLIC ${rclcpp_INCLUDE_DIRS})   #连接头文件（可执行文件名称 头文件存储路径）
target_link_libraries(ros2_cpp_node ${rclcpp_LIBRARIES})        #库文件连接（可执行文件名称 库文件存储路径）
```
- 04.在src同级目录处构建build文件夹，进入build目录，运用cmake指令
```
cmake ../
```
- 05.在build目录make,完成编译构建
```
make
```
- 06.在功能包的同级目录下构建功能包,在同级目录/build/first-cpp-pkg下产生可执行文件
```
colcon build
```
- 07.构建环境变量，指向这个pkg
```
source install/setup.bash
```
- 08.依赖声明（非必须）
在package.xml文件中声明包和节点的依赖关系

### C++导入第三方库方法
将第三方库资源文件下载到src/pkg_name/include目录下，然后在CmakeLists中添加关系
```
include_directories(include)  #包含include目录
```

### c++ qt库的使用
[qt代码文件](/08_ubuntu_ros2/05-chapt3_ws/topic_practice_ws/src/status_display/src/hello_qt.cpp)
[对应的CMakelist文件在这里](/08_ubuntu_ros2/05-chapt3_ws/topic_practice_ws/src/status_display/CMakeLists.txt)

# ros常用指令

```
ros2 node list
```
显示ros2正在进行的节点
```
ros2 node info </nodename>
```
显示ros2正在运行节点的信息

```
ros2 topic 
关于话题的指令相关

ros2 topic echo </nodename/话题名称>
显示话题的内容

ros2 topic info <话题名称>
显示话题的属性：几个订阅者、几个发布者、通讯类型

ros2 interface show <话题的type值>
查看消息接口的通讯方式

ros2 topic pub <话题名称> <通讯接口> <发布数据的内容>
发布一个话题

ros2 param list
查询参数列表

ros2 param describe <节点名> <参数名字>
查看参数的细节描述

ros2 param get <节点名> <参数名字>
获取参数的具体值

ros2 param set <节点名> <参数名字> <设置值>
设置具体参数的值

ros2 oaram dump <节点名> > <输出文件名.yaml>
将节点的参数值输出到文件

ros2 run <包名> <节点名> --ros-args --params-file <文件名.yaml>
使用文件中的单数运行节点



```
# ros中常用的代码
## c++版本
### 创建话题（发布话题与订阅话题）
```C++
class TurtleControlNode : public rclcpp::Node
{
private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_; // 声明发布者的智能指针
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr subscriber_;  // 声明订阅者的共享指针

public:
    explicit TurtleControlNode(const std::string &node_name) : Node(node_name)
    {
        publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10); // 实例化一个发布者
        subscriber_ = this->create_subscription<turtlesim::msg::Pose>("/turtle1/pose", 10, std::bind(&TurtleControlNode::on_pose_received_, this, std::placeholders::_1));
        //实例化一个订阅者<消息类型的接口>(订阅话题的名称，队列数量，回调函数)
    
    void on_pose_received_(const turtlesim::msg::Pose::SharedPtr pose) 
    {
        //每次受到话题之后要调用的回调函数。

    }
};
}
```
### 自定义消息接口方法
[系统状态显示消息接口在这里](/08_ubuntu_ros2/05-chapt3_ws/topic_practice_ws/src/status_interfaces/msg/)

1.构建功能包
```
ros2 pkg create status_interfaces --dependencies builtin_interfaces rosidl_default_genterators
构建自定义通讯接口的功能包，主要有两个依赖项，
builtin_interfaces，支持时间戳，
rosidl_default_genterators将ros同新街口转化为python和cpp源码的库。
```
2.删除无关目录/src、/install。在ws/src/pkg_name路径下新建文件夹msg。

> 注意：必须使用msg命名。

3.在msg目录下新建消息接口文件SystemStatus.msg文件

> 必须使用驼峰命名法，文件名以.msg作为后缀

4.在文件内部编写消息接口需要的数据类型。
```
builtin_interfaces/Time stamp   #记录时间戳
string host_name    #主机名称
float32 cpu_percent #cpu使用率
float32 memory_percent  #内存使用率
float32 memory_total #内存总大小
float32 memory_available    #可用内存总大小
float64 net_sent    #网络发送数据总量
float64 net_recv    #网络数据接受总量
```

5.在cmakelist文件中插入这样一个函数,可以同时插入多个消息接口文件
```
#这是一个cmake函数，来自于依赖rosidl_default_genterators，用于将msg等消息接口定义文件转换成库或者头文件类
rosidl_generate_interfaces(${PROJECT_NAME} 
"msg/SystemStatus.msg" 
"msg/SystemStatus.msg" 
"msg/SystemStatus.msg" 
DEPENDENCIES builtin_interfaces)
```

6.在package.xml文件中添加声明
```
  ///声明这是一个功能包文件。
  <member_of_group>rosidl_interface_packages</member_of_group>
```
会自动生成python和cpp版本的头文件

7.回到工作环境目录下，构建功能包并改变环境变量