
import rclpy
from status_interfaces.msg import SystemStatus #自定义的消息接口
from rclpy.node import Node
import psutil       #获取系统状态信息相关的库
import platform

'''
需求:获取系统状态信息,通过话题发布状态信息,使用自定义的SystemStatus消息接口
'''

'''
builtin_interfaces/Time stamp   #记录时间戳
string host_name    #主机名称
float32 cpu_percent #cpu使用率
float32 memory_percent  #内存使用率
float32 memory_total #内存总大小
float32 memory_available    #可用内存总大小
float64 net_sent    #网络发送数据总量
float64 net_recv    #网络数据接受总量
'''
class SysStatusPub(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.statu_publisher_ = self.create_publisher(
            SystemStatus, "sys_status", 10
        )
        self.timer_ = self.create_timer(1.0,self.timer_callback)
    
    def timer_callback(self):
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        net_io_counters = psutil.net_io_counters()

        msg = SystemStatus()
        msg.stamp = self.get_clock().now().to_msg() #直接将当前时间转化成一个时钟防盗信息里面。
        msg.host_name = platform.node()
        msg.cpu_percent = cpu_percent
        msg.memory_percent = memory_info.percent/1024/1024
        msg.memory_total = memory_info.total/1024/1024
        msg.memory_available = memory_info.available*1.0
        msg.net_sent = net_io_counters.bytes_sent/1024/1024
        msg.net_recv = net_io_counters.bytes_recv/1024/1024

        self.get_logger().info(f'发布：{str(msg)}')
        self.statu_publisher_.publish(msg)

def main():
    rclpy.init()
    node = SysStatusPub('sys_status_pub')
    rclpy.spin(node)
    rclpy.shutdown()