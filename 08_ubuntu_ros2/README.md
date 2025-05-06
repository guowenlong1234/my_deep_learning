# ubuntu基本操作
记录一下各种命令，方便以后重装
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
修改引导默认启动系统选项，记住自己的windows系统的序号，将GRUB_DEFAULT=0 改为 GRUB_DEFAULT=3,每次数10s后默认启动windows


