import rclpy
from rclpy.node import Node

def main():
    rclpy.init() #初始化节点
    node = Node("first_python_node")    #实例化一个node节点，节点名称为first_python_node
    node.get_logger().info("first_python_node is running")          #输出一个info级别的日志信息
    node.get_logger().warn("first_python_node is running")          #输出一个warn级别的日志信息
    rclpy.spin(node)    #运行节点，这句命令是阻塞的，不会退出。
    rclpy.shutdown()    #退出后进行清理节点


if __name__ == "__main__":
    main()
