cmake_minimum_required(VERSION 3.8)
project(cpp-learning)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
add_executable(cpp-learning-exe src/cpp-learn.cpp) 
add_executable(learn_auto_exe src/learn_auto.cpp) 
add_executable(learn-shared-ptr-exe src/learn-shared-ptr.cpp) 
add_executable(learn-lambda-exe src/learn-lambda.cpp) 
add_executable(learn-functional-exe src/learn-functional.cpp) 

ament_target_dependencies(cpp-learning-exe rclcpp) 

install(TARGETS cpp-learning-exe learn_auto_exe learn-shared-ptr-exe learn-lambda-exe learn-functional-exe
DESTINATION lib/${PROJECT_NAME}) #表示将可执行文件拷贝到install目录下指定位置。可以直接找到指定的可执行文件


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

