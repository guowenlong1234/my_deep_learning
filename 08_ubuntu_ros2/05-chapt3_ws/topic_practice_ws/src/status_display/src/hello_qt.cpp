/*
学习在cpp中使用qt界面展示模块
*/

#include <QApplication>
#include <QLabel>
#include <QString>

//导入qt相关库文件

int main(int argc, char* argv[])
{
    QApplication app(argc,argv);    //创建一个界面展示app
    QLabel* label = new QLabel();   //创建一个新的标签
    QString message = QString::fromStdString("hello qt!");  //创建一个string，是std的原生string类型
    label->setText(message);     //将消息内容添加到标签中
    label->show();              //将标签设置为显示
    app.exec();                  //执行应用，会在这里循环阻塞代码，直到关闭界面

    return 0;
}