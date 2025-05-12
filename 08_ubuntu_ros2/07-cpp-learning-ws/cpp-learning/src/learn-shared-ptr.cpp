#include <iostream>
#include <memory>

int main()
{
    auto p1 = std::make_shared<std::string>("This is a str.");
    //std::make_shared<数据类型/类>(参数)；返回值，对应类的共享指针 std::make_shared<std::string>类，写成auto

    std::cout<<"p1的引用计数:"<<p1.use_count()<<"指向的内存地址"<<p1.get()<<std::endl;//1
    auto p2 = p1;

    std::cout<<"p1的引用计数:"<<p1.use_count()<<"指向的内存地址"<<p1.get()<<std::endl;//2
    std::cout<<"p2的引用计数:"<<p2.use_count()<<"指向的内存地址"<<p2.get()<<std::endl;//2
    
    p1.reset();//释放引用
    
    std::cout<<"p1的引用计数:"<<p1.use_count()<<"指向的内存地址"<<p1.get()<<std::endl;//0
    std::cout<<"p2的引用计数:"<<p2.use_count()<<"指向的内存地址"<<p2.get()<<std::endl;//1

    std::cout<<"p2的指向内存地址的数据:"<<p2->c_str()<<std::endl ;   //调用成员方法

    return 0;
}
