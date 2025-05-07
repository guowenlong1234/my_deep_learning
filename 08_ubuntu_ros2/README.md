# 1.ubuntu基本操作
记录一下各种命令，方便以后重装
## 常用指令
修改环境变量指令
```
export PATH_NAME=PATH_VALUE
export RCUTILS_CONSOLE_OUTPUT_FORMAT=[{function_name}:{line_number}]:{message}
```
打印全部环境变量
```
printenv | grep PATH_NAME
```

## 常用环境变量
控制ros2中日志打印格式，

- RCUTILS_CONSOLE_OUTPUT_FORMAT=[{function_name}:{line_number}]:{message}
## 刚更新完成之后的操作

- 设置时钟

由于win与ubuntu时钟设置不同，修改ubuntu时钟方式为读取本地主板时间。
[设置系统时间看这里](https://www.bilibili.com/video/BV1554y1n7zv)

```
sudo apt install ntpdate
```
```
sudo ntpdate time.windows.com
```
```
sudo hwclock --localtime --systohc
```

- 修改启动引导项（解决开机黑屏无显卡驱动问题以及默认启动系统）

linux启动时，出现显卡驱动无法加载问题，多见于新显卡。解决方法禁用linux自带的第三方显卡驱动。第一次启动时在ubuntu启动项按e设置启动参数，在linux~~~~quiet splash ---这里改为quiet splash nomodesate在进行启动。进入之后将启动配置文件修改，配置文件在/etc/default/grub.#将GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"修改为
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nomodesate"。[开机启动项设置看这里](https://www.bilibili.com/video/BV1554y1n7zv)
```
sudo gedit /etc/default/grub
```
```
sudo update-grub
```
修改引导默认启动系统选项，记住自己的windows系统的序号，将GRUB_DEFAULT=0 改为 GRUB_DEFAULT=3,每次数10s后默认启动windows

# 2.关于v2ray的使用
下载链接：[v2ray客户端下载](https://github.com/selierlin/Share-SSR-V2ray/wiki/Linux%E4%BD%BF%E7%94%A8SS)

下载完成后在终端启动代理服务。注：其他软件想使用代理服务需要各自设置代理规则。默认代理端口为10808,可在客户端>设置>参数设置中修改

开启终端代理代码，可以写入./bashrc文件
```
export http_proxy=http://127.0.0.1:1080
```
```
export https_proxy=http://127.0.0.1:1080
```
```
export socks5_proxy=sockes5://127.0.0.1:1080
```
# 关于C++