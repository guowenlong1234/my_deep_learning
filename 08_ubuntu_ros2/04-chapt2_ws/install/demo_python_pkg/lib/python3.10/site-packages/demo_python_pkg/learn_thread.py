"""
这个功能包用来学习python多线程,下载小说并且统计字数
"""

import threading
import requests

class Download:
    def download(self, url,callback):
        print(f"线程：{threading.get_ident()} 开始下载：{url}")
        response = requests.get(url=url)
        response.encoding = 'utf-8'
        callback(url, response.text)      #调用回调函数
        

    def start_download(self, url, callback):
        #self.download(url, callback) 同步下载方法
        #接下来是多线程下载方法
        thread = threading.Thread(target=self.download, args=(url, callback))
        thread.start()
        

def world_count(url, result):
    """
    回调函数,要传入download函数内部用以计数
    """
    print(f"{url}:{len(result)}->{result[:10]}")

def main():
    download = Download()
    download.start_download("http://0.0.0.0:8000/novel1.txt", world_count)
    download.start_download("http://0.0.0.0:8000/novel2.txt", world_count)
    download.start_download("http://0.0.0.0:8000/novel3.txt", world_count)