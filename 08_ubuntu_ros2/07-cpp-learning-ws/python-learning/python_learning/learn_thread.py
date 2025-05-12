#python多线程学习
import threading
import requests

class DownLoad:
    def download(self, url, call_word_count):
        print(f"线程：{threading.get_ident()} 开始下载：{url}")
        response = requests.get(url)
        response.encoding = "utf-8"
        call_word_count(url, response.text)     #调用回调函数

    def start_download(self, url ,call_word_count):
        #self.download(url, call_word_count)这种调用是单线程的调用，会导致运行阻塞
        thread = threading.Thread(target=self.download,args=(url ,call_word_count))     #创建了一个新线程，作用对象是self.download函数，传入参数args=(url ,call_word_count）
        thread.start()      #新线程启动

def world_count(url, result):
    '''
    普通函数，用于回调
    '''
    print(f"{url}:{len(result)}->{result[:5]}")

def main():
    download = DownLoad()
    download.start_download("", world_count)
    download.start_download("", world_count)
    download.start_download("", world_count)