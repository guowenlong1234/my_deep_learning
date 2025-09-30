#include <iostream>
#include <functional> //函数包装器头文件

// 自由函数

void save_with_free_fun(const std::string &file_name)
{
    std::cout << "自由函数:" << file_name << std::endl;
}

class FileSave
{
private:
public:
    FileSave() = default;
    ~FileSave() = default;

    void save_with_member_fun(const std::string &file_name)
    {
        std::cout << "成员方法:" << file_name << std::endl;
    };
};

int main()
{

    FileSave file_save;

    // lambda函数

    auto save_with_lambda_fun = [](const std::string &file_name) -> void
    {
        std::cout << "Lambda函数:" << file_name << std::endl;
    };

    save_with_free_fun("file.txt");             // 调用自由函数
    file_save.save_with_member_fun("file.txt"); // 调用成员函数
    save_with_lambda_fun("file.txt");           //调用lambda函数

    //函数包装器的使用
    std::function<void(const std::string &)> save1 = save_with_free_fun;
    std::function<void(const std::string &)> save2 = save_with_lambda_fun;

    //成员函数放入包装器
    std::function<void(const std::string &)> save3 = std::bind(&FileSave::save_with_member_fun,&file_save,std::placeholders::_1); //bind绑定，参数分别是(函数指针、类指针、参数占位符)

    //将三种函数包装成了统一的函数类型

    save1("file.txt");
    save2("file.txt");
    save3("file.txt");

    return 0;
}