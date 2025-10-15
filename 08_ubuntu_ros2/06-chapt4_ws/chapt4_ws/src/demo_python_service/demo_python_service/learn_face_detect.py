'''
学习python中的人脸识别库,完成一个人脸识别的功能
'''
import face_recognition     #人脸识别库
import cv2  #openCV库
from ament_index_python.packages import get_package_share_directory #获取功能包share目录的绝对路径

def main():
    #1.获取图片的真实路径
    default_image_path = get_package_share_directory('demo_python_service') + '/resource/bus.jpg'
    print(f'图片的真实路径：{default_image_path}')

    #2.使用openCV打开图片
    image = cv2.imread(default_image_path)
    face_locations = face_recognition.face_locations(image, 1 ,"cnn")   #检测人脸

    #3.绘制人脸的边框
    for top,right,bottom,left in face_locations:
        cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),4)
    
    #4.结果显示
    cv2.imshow('Face Detece Result', image)
    cv2.waitKey(0)