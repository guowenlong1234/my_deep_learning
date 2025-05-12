#include <iostream>
#include <algorithm>    

int main()
{
    auto add = [](int a, int b) -> int {return a+b;};   //lambda表达式语法[捕获参数列表](定义参数列表) -> 返回值类型 {return 返回值}；
    int sum = add(200,50);

    auto print_sum = [sum]() -> void    //sum通过捕获参数列表捕获进来，不用传参，如果写[&]则表示将所有的上文参数全部捕获
    {
        std::cout<<sum<<std::endl;
    };

    print_sum();
    return 0;
}