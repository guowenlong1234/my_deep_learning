cmake_minimum_required(VERSION 3.8)
project(demo_cpp_topic)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(turtlesim REQUIRED)
#创建功能包时候命令行已经写了依赖项，这里自动已经加上了，如果内有是用命令行加入以来，需要在这里自己加入依赖

#接下来添加可执行文件
add_executable(turtle_circle src/turtle_circle.cpp)
add_executable(turtle_control src/turtle_control.cpp)

#添加头文件和依赖(可执行文件名字 依赖 依赖 )
ament_target_dependencies(turtle_circle rclcpp geometry_msgs)  
ament_target_dependencies(turtle_control rclcpp geometry_msgs turtlesim )




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

#将可执行文件添加到环境变量可以找到的位置
install(
  TARGETS turtle_circle turtle_control
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
