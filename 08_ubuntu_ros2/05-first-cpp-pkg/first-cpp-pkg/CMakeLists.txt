cmake_minimum_required(VERSION 3.8)   #指定版本
project(first-cpp-pkg)    #指定工程名称

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)       #直接查找到rclcpp头文件和库文件所在的路径，REQUIRED表示这个文件是必须的,会将他暂时存在类似于环境变量中头文件rclcpp_INCLUDE_DIRS等待调用

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
add_executable(first-cpp-pkg-exe src/first-cpp-pkg-node1.cpp)       #(生成的可执行文件名称 等待编译的文件名称)


message(STATUS ${rclcpp_INCLUDE_DIRS})  #头文件存储路径
message(STATUS ${rclcpp_LIBRARIES})  #库文件存储路径librclcpp.so 动态链接库

# target_include_directories(ros2_cpp_node PUBLIC ${rclcpp_INCLUDE_DIRS})   #连接头文件（可执行文件名称 头文件存储路径）
# target_link_libraries(ros2_cpp_node ${rclcpp_LIBRARIES})        #库文件连接（可执行文件名称 库文件存储路径）
ament_target_dependencies(first-cpp-pkg-exe rclcpp)   #这条命令用于代替上面两条命令（连接头文件和连接库文件）参数：（可执行文件名称 要连接的依赖库）

install(TARGETS first-cpp-pkg-exe DESTINATION lib/${PROJECT_NAME}) #表示将可执行文件拷贝到install目录下指定位置。可以直接找到指定的可执行文件


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
