import rclpy
from rclpy.node import Node
import requests

class NovePubNode(Node):
    def __init__(self,node_name):
        super().__init__(node_name)
        self.get_logger(),info(f'{node_name},启动！')
    
    def download(self, url):
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        callback(url)      #调用回调函数

def main():
    rclpy.init()
    node = NovePubNode("novel_pub")
    rclpy.spin(node)
    rclpy.shutdown()