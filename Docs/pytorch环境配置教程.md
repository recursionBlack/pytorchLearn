# **pytorch环境配置教程**

主要安装，

ubuntu 16.04 + (cuda + cuDNN) + Python3+pip3/Anaconda + Pytorch

## 1.启用window子系统及虚拟化

windows11专业版，需要开启虚拟化

#### 1.1 检查电脑是否已开启虚拟化支持

通过任务管理器查看：右键点击任务栏，选择 “任务管理器”，或按快捷键 Ctrl+Shift+Esc 打开任务管理器。切换到 “性能” 选项卡，选择 “CPU”，在右侧信息栏中查找 “虚拟化” 状态。若显示 “已启用”，表示虚拟化功能已开启；若显示 “已禁用”，则需进入 BIOS/UEFI 手动启用。

#### 1.2 使用图形界面开启虚拟化功能

1. 打开 **控制面版**
2. 访问 **程序和功能** 子菜单 **打开或关闭Windows功能**
3. 选择“适用于Linux的Windows子系统”与 “虚拟机平台”与“Hyper-V"
4. 点击“确定”
5. 重启

因为我们使用命令行已经执行，所以下边的打勾了，但hyper-v没有，所以我们要手动打勾，然后点击确定，最后重启。虚拟机平台只是hyper-v的部分功能，为了使用wsl2下的图形界面可视化，需要开启全部功能的hyper-v.

## 2.安装wsl2

#### 2.1 启用wsl2

Windows 11**通常自带**WSL 2的内核。

WSL 2是适用于Windows 11的Windows子系统的新版本，能在Windows系统上原生运行Linux二进制可执行文件。虽然系统自带了WSL 2内核，但默认情况下WSL是禁用的。用户可通过命令行或“Windows功能”来启用WSL 2，并下载安装喜欢的Linux发行版。

```
wsl --update
```

#### 2.2 设置默认WSL版本

我们只使用wsl2,power shell 以管理员方式运行

```
# 将 WSL 默认版本设置为 WSL 2
wsl --set-default-version 2
```

#### 2.3 配置linux分发版本

我们使用ubuntu,具体的可以是ubuntu20.04,选择你需要的版本即可。下载分发系统一定要用官方的，它是带有支持图形界面功能的。

```
Invoke-WebRequest -Uri https://aka.ms/wslubuntu2004 -OutFile Ubuntu.appx -UseBasicParsing
```

就能下载到Ubuntu.appx文件了

#### 2.4 安装到c盘

安装前，系统中没有安装其它系统。也可以用来检查系统中的所有子系统，非常常用

```
wsl -l -v
```

安装

```powershell
wsl.exe --install Ubuntu-20.04
```

把c盘的导出到D盘

```
wsl --export Ubuntu D:\WSL2\ubuntu.tar
```

导入盘的路径可以理解为安装的位置。可以看到文件夹下多一个ext4.vhdx文件夹：

在D盘导入

```
wsl --import Ubuntu D:\WSL2 D:\WSL2\ubuntu.tar --version 2
```

此时，电脑上就有两个子系统了，分别是在c盘的Ubuntu-20.04和在D盘的Ubuntu1，需要将C盘的注销掉

```
wsl --unregister Ubuntu-20.04
```

再次观察此时电脑上还有的子系统，同前面的一样

```
wsl -l -v
```

#### 2.5 默认系统设置及多系统选择

当有多个子系统启动时，带*花的是默认系统，当输入“wsl"，后会启动该系统，否则，我们要启同的同的系统需要指定：

```
wsl -d Ubuntu1
```

设置默认系统：

```
wsl --set-default Ubuntu1 # 或wsl -s Ubuntu1
```

#### 2.6 vhdx文件导入

有时重装系统或者想把ext4.vhdx文件拿到其它机器上使用，那么可以直接导入vhdx文件，具体命令是：

`wsl --import <导入Linux名称> <导入盘的路径>  版本(代表wsl2)`

```
wsl --import-in-place ubuntu2004 D:\WSL\ext4.vhdx --version 2
```

导出后，再次导入，会导致，原来的默认账户从用户变为了root。所以，需要进行修改

```powershell
ubuntu config --default-user your_username
```

导出后，再次导入，有时也会导致起始启动系统更改，默认的systemd，有时会被篡改为sysVinit

使用以下命令查看：

```
ps -p 1
```

#### 2.7 退出和关闭wsl

当进入到wsl2后，就像是进入了linux的命令行状态，如果想要退出的话，需要输入

```
exit
```

就能从wsl内退出到powershell了

但是，此时用

```
wsl -l -v
```

查看子系统，发现Ubuntu1此时的状态还是running，需要从外部进行强行关闭

```
wsl --shutdown
```

再次打开wsl，直接在powershell或者cmd里，输入

```
wsl
```

即可唤醒linux并进入

## 3.配置图形化显示桌面

此时，我们已经安装好了wsl和ubuntu20.04了，也能进入到命令行模式，还差使用linux gui了。可以参看一下官网[使用 WSL 运行 Linux GUI 应用 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/tutorials/gui-apps)。

看了几个后，才发现，原来这里安装的是，图形化应用啊，我最重要的图形化桌面哪里给我补啊？？？ 

所以这里，安装gnome，是linux的两大主流的桌面环境之一，另一个是KDE，KDE是qt开发的，更加美观。推荐kde，gnome太简洁了，跟windows桌面环境差距太大，容易让人不习惯。

```
sudo apt install KDE -y
```

我的个天呐！linux居然还有启动系统的分别？systemd和SysVinit，关键是，我的Ubuntu 20.04本来就应该是以systemd启动的啊，不知道为啥变成了sysVinit，还要改回来。

好像是改不回来的，因为是第二个系统，之前为了将linux从c盘搬出来，才发生了这种情况。把所有的Linux都卸载了，然后重新开一个，他就默认开在c盘，就正常了。

安装apt-fast代替原来的apt-get，可以多线程，更快的下载。

好吧，原来不需要安装linux桌面环境就能使用啊。我白白忙活一下午啊！！！

## 4.GPU使用

```
nvidia-smi
```



## 5.文件互访

#### 7.1直接访问wsl文件

这个还是说通过win系统操作wsl的文件，但不能把文件给拷出来，这个就相当于是个远程桌面。

```
sudo apt install nautilus
nautilus
```

### 7.2 wsl 访问 win

就在/mnt下，有c,d盘：

mnt就是传说中的挂载，可以使得内网机与宿主机的部分文件夹建立映射关系。

#### 7.3win访问wsl

打开文件资源管理器，有个小企鹅，就是，这个是我觉的最好的互访方法。

## 6.网络设置

正常情况下，直接使用是没有问题的，但在某些情况，比如本地windows系统中使用代理，这时代理对wsl不生效； 或都连接远程服务器，windows可以登陆但wsl 不可以；或者电脑双网卡，一个上内网，一个上外网，路由配置很困难等等，总的来说是配置前 windows和wsl 是两个ip,但配置后，wsl与windows基本是一样的，对windows生效的对wsl同样生效，或都在默认网络情况下不工作，可以试一下。

#### 6.1 配置.wslconfig
打开文件资源管理器，导航至 %UserProfile% 目录（通常是 C:\Users\你的用户名） 如果不存在，创建一个名为 .wslconfig 的新文件,随意一个文本编辑器都可以，加上以下内容：

```
[experimental]
autoMemoryReclaim=gradual  # 选择 gradual、dropcache 或 disabled
networkingMode=mirrored      # 设置为 mirrored 或 isolated
dnsTunneling=true            # 选择 true 或 false
firewall=true                # 选择 true 或 false
autoProxy=true               # 选择 true 或 false
sparseVhd=true               # 选择 true 或 false
```

#### 6.2 配置生效

重启wsl即可

```
wsl --shutdown
wsl -u your_user_name
```

## 7.远程映射开发

在 Windows 下使用 PyCharm 远程连接 WSL 中的 Anaconda 环境，可按以下步骤配置：

1. 确保 WSL 中 Anaconda 环境可用

   - 在 WSL 终端中验证 Anaconda 环境：

     ```bash
     conda activate your_env_name  # 激活环境
     which python  # 记录Python解释器路径（例如：/home/user/anaconda3/envs/your_env_name/bin/python）
     ```

2. 启用 WSL 的 SSH 服务

   在 WSL 中执行以下命令：

   ```bash
   # 安装OpenSSH服务器
   sudo apt update
   sudo apt install openssh-server
   
   # 启动SSH服务
   sudo service ssh start
   
   # （可选）设置开机自启
   sudo systemctl enable ssh
   ```

3. 在PyCharm中配置远程解释器

   1. **打开 PyCharm 项目** → **File** → **Settings**（Windows/Linux）或 **PyCharm** → **Preferences**（macOS）。

   2. **Project: [项目名]** → **Python Interpreter** → 点击齿轮图标 → **Add**。

   3. 在弹出窗口中选择WSL Interpreter：

      - **Distribution**：选择已安装的 WSL 发行版（如 Ubuntu）。

      - **Interpreter path**：填写 WSL 中 Python 解释器的路径（例如：`/home/user/anaconda3/envs/your_env_name/bin/python`，本机应为：

        `/home/kuan/anaconda3/envs/torch_gpu/bin/python3.13`）。

      - 点击 **OK** 测试连接。

4. 验证配置

   - PyCharm 会自动同步 WSL 环境中的 Python 包，你可以在 **Python Interpreter** 设置中查看已安装的包（如`torch`、`torchvision`）。

   - 创建一个简单的测试脚本，确保远程环境正常工作：

     ```python
     import torch
     print(torch.__version__)
     print(torch.cuda.is_available())  # 应输出True
     ```

遇到的问题：

pycharm中的linux Distribution中的名称无法更改，只能是Ubuntu，而我装的是Ubuntu-20.04，导致名称对不上，而且无法从pycharm上更改，最后只能修改WSL 发行版名称。

```powershell
# 导出当前 WSL 配置
wsl --export Ubuntu-20.04 C:\temp\ubuntu-20.04.tar

# 注销当前发行版（数据不会丢失）
wsl --unregister Ubuntu-20.04

# 使用新名称重新导入
wsl --import Ubuntu C:\wsl\Ubuntu C:\temp\ubuntu-20.04.tar --version 2

# 删除临时文件
Remove-Item C:\temp\ubuntu-20.04.tar
```



## 8.安装anaconda或miniconda

打开WSL终端

在官网[Installing Miniconda - Anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install)找到下载链接，用wget下载.sh文件

```
wget 下载链接
```

之后运行.sh文件，安装anaconda或miniconda

#### 8.1 安装miniconda

1.Run the following four commands to download and install the latest Linux installer for your chosen chip architecture. Line by line, these commands:

- create a new directory named “miniconda3” in your home directory.
- download the Linux Miniconda installation script for your chosen chip architecture and save the script as `miniconda.sh` in the miniconda3 directory.
- run the `miniconda.sh` installation script in silent mode using bash.
- remove the `miniconda.sh` installation script file after installation is complete.

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

2.After installing, close and reopen your terminal application or refresh it by running the following command:

```
source ~/miniconda3/bin/activate
```

3.Then, initialize conda on all available shells by running the following command:

```
conda init --all
```

If you don’t initialize conda after installation, you might see a “conda not found” error, even though conda is installed. See the [Conda: command not found on macOS/Linux](https://www.anaconda.com/docs/reference/troubleshooting/#conda-cmd-not-found) troubleshooting topic for possible solutions.

退出miniconda

```
conda deactivate
```



#### 8.2 卸载miniconda

由于miniconda是精简版，而anaconda是完全体，懒人版，新手更适合用anaconda，所以要把miniconda卸载了，安装anaconda

首先，先卸载已安装的环境（base）及其文件夹

(Optional) If you have created any environments outside your `miniconda3` directory, Anaconda recommends manually deleting them to increase available disc space on your computer. *This step must be performed before uninstalling Miniconda*.

Uninstall environments outside the miniconda3 directory

1.View a list of all your environments by running the following command:

```
conda info --envs
```

If you have any environments in a directory other than miniconda3, you need to uninstall the directory that contains the environments. a.

Uninstall the directory by running the following command:

```
# Replace <PATH_TO_ENV_DIRECTORY> with the path to the directory that contains the environments
~/miniconda3/_conda constructor uninstall --prefix <PATH_TO_ENV_DIRECTORY>
~/miniconda3/_conda constructor uninstall --prefix /home/kuan/miniconda3
```

再卸载miniconda

1.Open a new terminal application window.

2.Deactivate your `(base)` environment by running the following command:

```
conda deactivate
```

You should no longer see `(base)` in your terminal prompt.

As of Miniconda v24.11.1, an `uninstaller.sh` script is available to help you remove Miniconda from your system. Run the basic script to remove Miniconda and its shell initializations, or add arguments to remove additional user or system files. If your version does not have the uninstaller script, use the instructions under Manual uninstall.

If you have installed Miniconda into a system location, you must use `sudo -E` to run the uninstaller.

For example, the `.pkg` installer for macOS installs Miniconda into a system location, `/opt/miniconda3`.

4.Close and reopen your terminal to refresh it. You should no longer see (base) in your terminal prompt.

最后，通过

```
conda info --envs
```

检查是否还有miniconda

#### 8.4 安装anaconda

1.Download the latest version of Anaconda Distribution by opening a terminal and running one of the following commands (depending on your Linux architecture):

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
```

2.(Recommended) Verify the integrity of your installer to ensure that it was not corrupted or tampered with during download.

3.Install Anaconda Distribution by running one of the following commands (depending on your Linux architecture):

```
bash ~/Anaconda3-2025.06-0-Linux-x86_64.sh
```

4.Press Return to review the [Anaconda’s Terms of Service (TOS)](https://anaconda.com/legal). Then press and hold Return to scroll.

5.Enter `yes` to agree to the TOS.

6.Press Return to accept the default install location (`PREFIX=/Users/<USER>/anaconda3`), or enter another file path to specify an alternate installation directory. The installation might take a few minutes to complete.

7.Choose an initialization options:

- Yes - `conda` modifies your shell configuration to initialize conda whenever you open a new shell and to recognize conda commands automatically.
- No - `conda` will not modify your shell scripts. After installation, if you want to initialize, you must do so manually. For more information, see [Manual shell initialization](https://www.anaconda.com/docs/getting-started/anaconda/install#manual-shell-init-linux).

8.The installer finishes and displays, “Thank you for installing Anaconda3!”

9.Close and re-open your terminal window for the installation to fully take effect, or use the following command to refresh the terminal, depending on your shell:

```
source ~/.bashrc
```

退出anaconda

```
conda deactivate
```

检查conda环境

```
conda info --envs
```

再次进入到linux里面，发现直接带了个(base)???看来一进来，就进入到了anaconda的环境里了。

不过还是可以手动退出和进入conda环境里的。

## 9.conda创建虚拟环境，安装pytorch+cu116

在conda prompt中执行如下命令，创建虚拟环境

```bash
conda create -n torch_gpu python=3.13
```

在pytorch官网找到对应命令，在虚拟环境下执行即可（根据自己的需要，找版本，安装的cuda版本不能高于机器中CUDA Version。我的机器CUDA Version是12.3，所以安装的cu116（cuda 11.6）没毛病）

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在wsl中，如果某个链接，下载的特别慢，该如何加速？

豆包：

在WSL中遇到下载链接速度慢的情况，可以通过更换软件源、优化网络配置、利用第三方下载工具等方法来加速，具体如下：

- **更换软件源**：WSL中Linux发行版默认的官方源可能因地理位置等原因下载缓慢，可更改为国内镜像站点。以Ubuntu为例，打开WSL终端，使用命令`sudo sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list`将APT源改为清华大学开源软件镜像站，然后执行`sudo apt update`更新APT缓存，即可提高下载效率。 
- **优化网络环境**：确保已连接至稳定且带宽充足的网络，关闭其他占用大量带宽的应用程序和服务。如果使用无线网络，尽量靠近路由器以获得更好的信号强度。另外，也可以尝试更换DNS服务器，如使用Google DNS（8.8.8.8）或阿里云DNS（223.5.5.5）。 
- **启用大型数据包发送功能**：“vEthernet(WSL)”虚拟网卡若未开启Large Send Offload版本选项，可能影响TCP/IP性能。可在PowerShell中使用`Get - NetAdapterBinding - IncludeHidden - Name "vEthernet (WSL)"`和`Get - NetAdapterAdvancedProperty - IncludeHidden - Name "vEthernet (WSL)"`命令查询现有设置状况，若未开启则使用`Set - NetAdapterAdvancedProperty - IncludeHidden - Name "vEthernet (WSL)" - DisplayName "Large*" - DisplayValue Enabled`命令启动LSO v2支持。 
- **利用第三方下载工具**：可以在WSL中安装aria2等下载工具，利用其多线程功能加速下载。先使用`sudo apt install aria2`命令安装aria2，然后使用类似`aria2c -x 16 your_download_link`的命令进行下载，其中`-x 16`表示使用16个线程下载，可根据实际情况调整线程数。 
- **使用离线安装包**：对于一些软件或更新，可以从官方GitHub发布页面等渠道下载离线安装包，如WSL的更新包可从https://github.com/microsoft/WSL/releases下载，然后手动安装，避免在线下载的速度瓶颈。 
- **配置代理**：如果Windows主机已使用代理，可在WSL中配置使用相同的代理。先获取Windows主机IP，然后设置临时代理，假设Windows代理端口为7890，可执行`export http_proxy="http://主机IP:7890" export https_proxy="http://主机IP:7890"`命令（需根据实际情况替换主机IP和端口），之后再进行下载操作。

实际执行：利用第三方工具，aria2

```bash
sudo apt install aria2
```

```bash
aria2c -x 16 -s 16 -k 1M "https://download.pytorch.org/whl/cu128/nvidia_cudnn_cu12-9.7.1.26-py3-none-manylinux_2_27_x86_64.whl"
```

在下载了一个1g多的包，用了两个小时后，没想到后面还有一个700M的包，我是在是忍受不了了，最终选择，用了加速方式，把这个700M的给下了，并行下载后，下载速度果然达到了

后面发现还是差一个，就查看清华源的。

好吧，清华源也被干掉了

#### 换成11.8版本的

好吧，换成11.8快的要死。原来是12.8太先进了，所以就没法下载了啊

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

我去，用11.8版本后，果然很快就下载好了

我这一上午白折腾了啊

#### 测试pytorch环境

进入WSL终端，激活你用conda创建的pytorch环境，执行

```python
import torch # 导入pytorch包
print(torch.cuda.is_available()) # pytorch能否使用NVIDIA显卡，应输出True
print(torch.cuda.device_count()) # 可用的显卡数量，应该大于0
```

到这一步，你已经可以使用WSL搞深度学习了

## 10.安装opencv

找到自己的conda环境

```bash
conda install -c conda-forge opencv
```

