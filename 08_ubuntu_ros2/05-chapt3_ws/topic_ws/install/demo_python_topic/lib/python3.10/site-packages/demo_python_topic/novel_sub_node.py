'''
节点功能，订阅话题，并将小说内容合成语音
技术1:如何订阅话题。self.novel_subscriber = self.create_subscription(String, 'novel',self.novel_callback, 10)
技术2:如何合成发布语音。speaker = espeakng.Speaker()
        speaker.voice = 'zh'
技术3:小说来的太快播放 太慢怎么办。防止到队列中，在开启一个新线程进行朗读。
'''

import espeakng
import rclpy
from rclpy.node import Node
from example_interfaces.msg import String
from queue import Queue             #python中队列库的队列类，使用队列来进行小说的组织
import threading                    #多线程相关库
import time

class NovelSubNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.novel_quene = Queue()
        self.get_logger().info(f'{node_name},启动！')
        self.novel_subscriber = self.create_subscription(String, 'novel',self.novel_callback, 10)
        self.speech_thread = threading.Thread(target=self.speak_thread)     #创建一个线程，线程运行函数self.speak_thread进行朗读。
        self.speech_thread.start()
    
    def novel_callback(self,msg):
        self.novel_quene.put(msg.data)

    def speak_thread(self):
        speaker = espeakng.Speaker()
        speaker.voice = 'zh'

        while rclpy.ok():   #检测ROS上下文是否OK
            if self.novel_quene.qsize()>0:
                text = self.novel_quene.get()
                self.get_logger().info(f'{text}')
                speaker.say(text)
                speaker.wait()  #等他说完
            else: #如果已经没有可读内容，让当前线程休眠
                time.sleep(1)   #休眠1秒
    
def main():
    rclpy.init()
    node = NovelSubNode('novel_sub')
    rclpy.spin(node)
    rclpy.shutdown()