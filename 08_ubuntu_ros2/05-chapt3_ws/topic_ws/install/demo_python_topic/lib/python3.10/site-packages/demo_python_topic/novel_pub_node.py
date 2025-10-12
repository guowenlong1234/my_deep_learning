"""
任务需求：
1.下载小说,并且通过一个话题每隔5秒逐行发布小说。

实现技术：
创建队列，创建话题发布者，下载小说，将小说装载进入队列，话题发布者从队列中取出小说的每一行，进行发布。
定时方法采用ros2自带的create_timer(5,self.timer_callback)方法。不许使用回调函数的方法。
"""



import rclpy
from rclpy.node import Node
import requests
from example_interfaces.msg import String
from queue import Queue     #python中队列库的队列类，使用队列来进行小说的组织
class NovePubNode(Node):
    def __init__(self,node_name):
        super().__init__(node_name)
        self.get_logger().info(f'{node_name},启动！')
        self.novels_quene = Queue()     #创建一个python队列类型
        self.novel_publisher = self.create_publisher(String, 'novel',10)   #python创建话题发布者的方法，第一个参数为接口类型，第二个参数为发布话题名称，第三个参数为服务质量，指话题发布信息缓存数量。
        self.create_timer(5,self.timer_callback)     #创建python的计时器，方法从父类中继承而来.两个必传参数，第一个浮点数表示每隔多久运行一次，第2个参数为callback回调函数。


    def timer_callback(self):
        if self.novels_quene.qsize()>0:
            line = self.novels_quene.get()
            msg = String()
            msg.data = line
            self.novel_publisher.publish(msg)
            self.get_logger().info(f'发布了：{msg.data}')

    def download(self, url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        text = response.text
        self.get_logger().info(f'下载{url},{len(text)}')
        for line in text.splitlines():
            self.novels_quene.put(line) #把小说按照每一行放进队列中存储。


def main():
    rclpy.init()
    node = NovePubNode("novel_pub")
    node.download("http://0.0.0.0:8000/novel1.txt")
    rclpy.spin(node)
    rclpy.shutdown()