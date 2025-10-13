cmake_minimum_required(VERSION 3.8)
project(status_display)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(status_interfaces REQUIRED)

#查找qt5库
find_package(Qt5 REQUIRED COMPONENTS Widgets)

#添加可知性文件
add_executable(hello_qt src/hello_qt.cpp)
add_executable(sys_status_display src/sys_status_display.cpp)

#连接依赖库，注意这里不能使用ros提供的工具，只能使用cpp原本的连接工具。
target_link_libraries(hello_qt Qt5::Widgets)
target_link_libraries(sys_status_display Qt5::Widgets)

#连接ros2的依赖库，需要用ros2的方法进行。
ament_target_dependencies(sys_status_display rclcpp status_interfaces)




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

install(TARGETS hello_qt sys_status_display
DESTINATION lib/${PROJECT_NAME}) #表示将可执行文件拷贝到install目录下指定位置。可以直接找到指定的可执行文件

ament_package()
