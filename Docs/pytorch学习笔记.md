# Pyorch学习笔记

我的天呐！现在这些视频教学都没有对应的教学文档了吗？？？还要自己写吗？关注了几个公众号，最后发现还是引流卖课的，教学ppt是真的不给啊。也可能确实是因为技术太新了，人家up还要卖课呢，所以只能自己手写了。

## 第3节.环境配置

ubuntu 16.04 + (cuda + cuDNN) + Python3+pip3/Anaconda + Pytorch

### 安装步骤1 ubuntu 16.04 

尽量安装双系统，不要用虚拟机，

安装双系统教程：

也可以用wsl2+linux gui界面

安装显卡驱动：

```
https//www.imooc.com/articla/303674
```

nvidia-smi

### 安装步骤2 --- CUDA/cuDNN

1.CUDA10

2.cuDNN

3.教程：

```
https//www.imooc.com/articla/303675
```

### 安装步骤3 --- Python3 OR Anaconda

1.课程中推荐使用Python3 + pip3安装所需依赖包

2.Anaconda安装源：

```
https://mirros.tuna.tsinghua.edu.cn/anaconda/archive/
```

### 安装步骤4 --- Pytorch

1.网址：

```
https://pytorch.org/get-started/locally/
```

2.GPU版本：

```
pip3 install torch==1.5.0+cu101 torchvison==0.6.0+cu101-f https://download.pytorch.org/whl/torch_stable.html
```

3.CPU版本：

```
pip3 install torch==1.5.0+cu101 torchvison==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 吐槽

我操！pytorch环境安装好复杂啊！堪比我当初安装vscode或vs后，配置环境了。估计要整一晚上+一上午了吧。好多资源都要关注公众号才给，卖课引流啊

### anaconda环境

什么叫anaconda环境？

Anaconda环境是一个由Anaconda发行版提供的虚拟环境，它能够对Python版本以及各类第三方包进行有效管理。借助Anaconda环境，你可以在同一台计算机上创建多个相互独立的环境，每个环境都能拥有独特的Python版本和包配置，这样一来，不同项目之间就不会因为依赖冲突而产生问题。 

#### 主要功能 

1. **环境隔离**：能够创建多个独立的环境，防止不同项目的依赖出现冲突。 
2. **包管理**：可以便捷地安装、更新或者卸载包，并且能自动解决依赖关系。
3.  **Python版本管理**：支持同时安装和使用不同版本的Python。 

#### 基本操作 

下面为你介绍一些常用的命令： 

```
```bash 
# 创建一个名为myenv的新环境，指定Python版本为3.9 
conda create -n myenv python=3.9 
# 激活环境 
conda activate myenv 
# 安装包（例如numpy） 
conda install numpy 
# 查看已安装的包 
conda list 
# 导出环境配置 
conda env export > environment.yml 
# 从配置文件创建环境 
conda env create -f environment.yml 
# 停用当前环境 conda deactivate 
# 删除环境 
conda env remove -n myenv 
```
```

#### 应用场景 

- **多项目开发**：当你同时进行多个项目的开发，而这些项目对同一包的版本要求不同时，Anaconda环境就能很好地解决这个问题。
-  **试验环境**：你可以创建临时性的环境来尝试新的包或者Python版本，而不用担心会对现有环境造成影响。 
- **协作开发**：通过导出环境配置文件，能够确保团队成员的开发环境保持一致。 如果你需要进一步了解如何创建或者管理Anaconda环境，请随时告诉我！

anaconda创建环境后，如何切换进新环境里？

在使用 Anaconda 创建新环境后，你可以使用以下命令切换进入新环境： 

#### 激活环境 

```# Windows系统 ```

```conda activate your_env_name ```

 ```# macOS/Linux系统 ```

```source activate your_env_name  ```

 ```# 适用于较旧的conda版本 ```

```conda activate your_env_name  ```

```# 适用于较新的conda版本 ``` 

#### 验证环境 

激活后，可以通过以下命令确认当前使用的环境：

```bash 
# 查看当前环境路径
which python	# macOS/Linux
where python 	# Windows
# 查看所有环境（当前环境会被星号*标记）
conda info --envs
```

####  退出环境

如果需要退出当前环境，回到基础环境，可以使用： 

```bash
conda deactivate
```

### 完成了

卧槽，历时两天！0720，终于把这个狗日的pytorch给安好了，太麻烦了。终于可以继续学习了！

------

## 第5节.机器学习中的分类与回归问题

样本，模型，训练，推理，测试

y = f(x)

样本：包含属性和标签，属性是x，标签是y，

模型是f

f(x) = wx +b

训练就是获取w和b的过程

测试就是评价我们模型的过程，评价指标和参数，模型性能评估

推理，就是根据属性，和模型，计算出标签的过程，也就是最终的应用过程

## 第6节.Pytorch的基本概念

### Tensor

张量。标量，单个数字。向量，一维数组。矩阵,二维数组。张量，任意维度数组，张量是任意维度的数组，维度数量称为 “阶数（Rank）”。它可以是 0 维（标量）、1 维（向量）、2 维（矩阵），甚至更高维（如 3 阶、4 阶等）。
0 阶张量：标量（如单个数字 5）
1 阶张量：向量（如
[1,2,3]
）
2 阶张量：矩阵（本质上是张量的特例）
3 阶张量：可理解为 “矩阵的堆叠”，例如一张彩色图片（宽 × 高 ×RGB 通道）就是 3 阶张量。

### variable

变量，也就是参数

### nn.Module

啊

## 第7节. Tensor的基本概念

 tesnsor可以和numpy相互转换

### Tensor的类型

和编程语言的数据类型很像

### Tensor的创建

定义尺寸式，拷贝构造，全1Tensor，全0Tensor，单位Tensor，切片式，随机的，均匀标准分布，正态分布，均匀分布，等类型

## 第9节.Tensor的属性

每个Tensor有torch.dtype、torch.device、torch.layout三种属性

torch.device标识了torch.Tensor对象在创建之后所存储在的设备名称。cpu，gpu(cuda)

torch.layout表示torch.Tensor内存布局的对象

### 稠密（dense）的张量和稀疏（sparse）的张量

稠密的张量日常用的，对应于内存中，一块连续的区域

### 稀疏（sparse）的张量

torch.sparse_coo_tensor

coo类型表示了非零元素的坐标形式

稀疏的好处，表达了我们当前的数据，非零元素的数量，张量中零元素越多，张量 的计算越简单

稀疏可以使模型变得非常的简单

减少在内存中的开销

## 第10节.Tensor的算术运算

四则运算，其他运算，

### 加法运算

4种方法，其中第四种add_类似于+=

### 减法

也是4种，

### 乘法

对应元素相乘，类似于矩阵里的点乘，在张量里叫做哈达玛积。也是有4种方法

### 除法

也是有4种方法

### 矩阵运算

5种实现方式，torch.mm()、torch.matmul()、@

对于高于2维的Tensor，定义其矩阵乘法，仅在最后的两个维度上，要求前面的维度必须保持一致。就像矩阵的索引一样，并且运算操作只有torch.matmul().

### 幂运算

也是有4种方法

最常用的是e的n次方

### 开方运算sqrt()

3种方法

### 对数运算

log

也是4种

## 13. Pytorch中的in-place操作

也称原位操作，不允许使用临时变量，比如+=，等

### Pytorch中的广播机制

张量参数可自动扩展为相同大小

广播机制需要满足两个条件：

- 每个张量至少有一个维度
- 满足右对齐
- torch.rand(2, 1, 1) + torch.rand(3)

## 14.取整和取余运算

aaaa

## 15.Tensor的比较运算

```
eq, equal, ge, gt, le, lt, ne
```

### 排序

```
sort, topk, kthvalue
```

### 有界性判断

```
isfinite, isinf, isnan
```

返回一个mask张量

## 16.Tensor的三角函数

sin, cos, tan

## 17.Tensor中的其他的数学函数

```python
abs()		# 绝对值
sign()		# 符号函数，分段函数形式, 分类问题
sigmoid()	# 当x趋近于无限小时，y趋近于0，
# 当x趋近于无限大时，y趋近于1.sign()连续性的逼近，激活函数
```

## 18.Tensor中的统计学相关函数

数据分析时调用这些函数，直方图。灰度直方图

传统机器学习，特征统计

## 19. Tensor的torch.distributions

distributions包含可参数化的概率分布和采样函数

得分函数：强化学习种策略梯度方法的基础，一种反馈机制，在强化学习中用的较多

pathwise dericative估计器：变分自动编码器种的重新参数化技巧

计算机视觉中，应用的不算特别多。简单的过一下

## 20.Tensor中的随机抽样

定义随机种子：torch.manual_seed(seed)

定义随机数满足的分布：torch.normal()

## 21.Tensor的范数运算

范数：在泛函分析中，它定义在赋范线性空间中，并满足一定的条件，即1：非负性，2：齐次性，3：三角不等式

常被用来度量某个向量空间（或矩阵）中的每个向量的长度或大小。比如，模长，distance

0范数，1范数，2范数，p范数，核范数

torch.dist(input, other, p=2)计算p范数

torch.norm()计算2范数

范数在机器学习中的作用

定义loss

参数约束

## 22.Tensor的矩阵分解

常见的矩阵分解：

LU分解，QR分解，EVD分解，SVD分解

以特征值分解和奇异值分解为主

特征值分解与PCA降维

奇异值分解与LDA算法，与特征工程，引入学习

### 特征值分解

将矩阵分解为由其特征值和特征向量表示的矩阵之积的方法

特征值vs特征向量：

Av = lambda v

A是矩阵，v是特征向量，lambda是特征值

A = QEQ逆

E（西格玛）是对角线为特征值序列，其余全为0的矩阵

A一定是一个nxn的方阵

### PCA与特征值分解

PCA：主成分分析，将n维特征通过投影矩阵，降维映射到k维上，这k维是全新的正交特征，也被称为主成分，是在原有n维特征的基础上，重新构造出来的k维特征

PCA算法的优化目的就是：

- 降维后同一维度的方差最大。特征越丰富

- 不同维度之间的相关性为0。信息冗余度尽可能的低

- 协方差矩阵，描述特征的方差和相关性

 ## 23.Tensor中的矩阵分解2

奇异值分解：svd分解

A = UEV转置

A是mxn矩阵，

U是mxm矩阵，

E(西格玛)是mxn矩阵

V转置是nxn矩阵

svd分解是奇异值分解的特例，当U和V刚好是同阶互逆的时候，奇异值分解就是svd分解了。类似于正方形是特殊的长方形

### LDA（鉴别分析或叫判别分析）与奇异值分解

同类物体之间的间距尽可能小

不同类物体之间的间距尽可能大

通过优化函数来表示同类之间的距离

优化函数=类间/类内

Pytorch中的奇异值分解方法：torch.svd()

## 24.Tensor的裁剪运算

对Tensor中的元素进行范围过滤

常用于梯度裁剪（gradient clipping），即在发生梯度离散或梯度爆炸时对梯度的处理

a.clamp(2, 10)

## 25.Tensor的索引与数据筛选

```
where
gather
index_select
mask_select
take
nonzero
```

## 26.Tensor的组合/拼接

```
cat
stack
gather
```

## 27.Tensor的切片

```
chunk
split
```

## 28.Tensor的变形操作

```
reshape
t	# 转置
transpose 	# 交换两个维度
squeeze		# 去除那些维度大小为1的维度
flip		# 给指定维度反转张量
rot90		# 旋转一定的角度
```

## 29.Tensor的填充操作

定义Tensor，并填充指定的数值，全1矩阵，全0矩阵，全n矩阵

```
full
```

## 30. Tensor的频谱操作

计算机视觉上基本上用不到的，主要用在了一些云信号的解析，所以这里就不详细说了

时域信号，转化为频域信号的操作

```
fft
ifft
rfft
irfft
stft
```

## 31.模型的保存/加载

```python
save
load
```

### 并行化

```python
get_num_threasd()	# 获得用于并行化CPU操作的OpenMP线程数
set_num_threasd(int)	# 设定用于并行化CPU操作的OpenMP线程数
```

### 分布式

python在默认情况下只使用一个GPU，在多个GPU的情况下，就需要使用pytorch提供的DataParallel

单机多卡，

多机多卡

### Tensor on GPU

用方法to()可以将Tensor在CPU和GPU（需要有硬件）之间互相移动

### Tensor的相关配置

；

### Tensor与numpy的互相转换

```
torch.from_numy(ndarry)
a.numpy()
```

安装opnecv

```
conda install -c conda-forge opencv
```

## 32.Variable&Autograd

### 什么是导数？

导数（一元函数）是变化率，是切线的斜率，是瞬间速度

### 什么是方向导数？

函数在A点无数个切线的斜率的定义。每一个切线都代表一个变化的方向

面上一个点的切面

### 什么是偏导数

多元函数降维时候的变化，比如：二元函数固定y，只让x单独变化，从而看成是关于x的一元函数的变化来研究

### 什么是梯度

函数在A点无数个变化方向中，变化最快的那个方向

记为：的尔特f或者gradf

## 33.梯度与机器学习中的最优解

有监督学习（LDA，svm(随机森林)，深度学习），无监督学习（聚类，PCA），半监督学习

样本X，标签Y

有标签的学习，叫有监督的，没标签的，纯靠样本之间关联性推理的，叫无监督学习

标签是不准确（弱标签），甚至出错（伪标签），一部分样本有标签，一部分没有。

半监督最适合学术上找创新点，写论文

```
Y = f(X)
Y = f(w, X)
```

希望找到一组W，使得X尽可能的接近Y

```
loss = ||f(w, X) - Y||2
argmin loss
```

## 34.Variable is Tensor

目前Variable 已经与Tensor合并

每个tensor通过requires_grad来设置是否计算梯度

用来冻结某些层的参数

## 35.如何计算梯度

链式法则：两个函数组合起来的复合函数，导数等于里面函数带入外函数值的导乘以里面函数之导出

```
z = f(g(x))
```

## 36.关于Autograd的几个概念

叶子张量

grad:该Tensor的梯度值，每次在计算backward时都需要将前一时刻的梯度归零，否则梯度值会一直累加

 grad_fn：叶子节点通常为None，只有结果节点的grad_fn才有效，用于指示梯度函数是那种类型

backward函数：

## 37.关于Autograd的几个概念2

torch.autograd.Function

每一个原始的自动求导的运算，实际上是两个在Tensor上运行的函数

- forwoard函数计算从输入Tensors获得的输出Tensors
- backward函数接收输出Tensors对于某个标量值的梯度，并且计算输入Tensors相对于该相同标量值的梯度
- 最后，利用apply方法执行相应的运算
  - 定义在Function类的父类_FunctionBase中定义的一个方法

## 38.torch.nn库

torch.nn是专门为神经网络设计的模块化接口

nn构建于autograd之上，可以用来定义和运行神经网络

```python
nn.Parameter
nn.Linear & nn.conv2d
nn.functional
nn.Module
nn.Sequential
```

### nn.Parameter

神经网络，一般被称为模型；包括结构和参数

- 定义可训练参数
- self.my_param = nn.Parameter(torch.randn(1))
- self.register_parameter
- nn.ParameterList & nn.ParameterDict

### nn.Linear & nn.conv2d

- 各种神经网络层的定义，继承于nn.Module的子类
  - self.conv1 = nn.Conv2d(1, 61 (5, 5))
  - 调用时：self.conv1(x)

- 参数为parameter类型
  - layer = nn.Linear(1,  1)
  - layer.weight = nn.Parameter(torch.FloatTensor([[0]]))
  - layer.bias = nn.Parameter(torch.FloatTensor([0]))

### nn.functional

- 包括torch.nn库中所有函数，包含大量loss和activatin function
- nn.functional.xxx是函数接口
- nn.functional.xxx无法与nn.Sequential结合使用
- 没有学习参数的等根据个人选择使用
- 需要特别注意dropout层。训练时有用，推理时无用

### nn与nn.functional有什么区别？

- nn.functional.xxx是函数接口
- nn.Xxx是.nn.functional.xxx的类封装，并且nn.Xxx都继承于一个共同祖先nn.Module
- nn.Xxx除了具有nn.functional.xxx功能之外，内附带nn.Module相关的属性和方法，eg.train()，eval(), load_state_dict, state_dict
- 搭建神经网络时，尽量使用nn.Xxx

### nn.Sequential

将几个卷积层，串联起来，作为一个序列

### nn.ModuleList

定义模型, 也叫网络结构。需要定义init和forward

### nn.ModuleDict

定义模型, 也叫网络结构

### nn.Module

- 它是一个抽象概念，即可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络

- model.parameters()

  可训练的参数

- model.buffers()

  不可训练参数

- model.state_dict()

  访问当前网络中所有的参数，并保存这些参数，之后可以加载这些参数

- model.modules()

  定义了我们当前的模型中，到底包含了那些模块，

- forward(), to()

  前向推理。把资源从cpu搬到gpu里

- https://pytorch.org/docs/stable/nn.html#torch.nn.Module

### Parameters vs buffers

- 一种是反向传播需要被optimizer更新的，称之为parameter

  - ```
    self.register_parameter("param, param")
    ```

    ```
    self.param = nn.Parameter(torch.randn(1))
    ```

- 一种是反向传播，不需要被optimizer更新，称之为buffer

  - ```
    self.register_buffer("my_buffer", torch.randn(1))
    ```

### state_dict()  &  load_state_dict

- ```
  torch.save(obj=model.state_dict(), f="models/net.pth")
  ```

- ```
  model.load_state_dict(torch.load(models/net.pth))
  ```

- 完成对模型的保存和加载，保存中间结果

## 39.Visdom介绍

Facebook专门为Pytorch开发的一款可视化工具，开源于2017年3月，提供了大多数的科学运算可视化的api：

- https://github.com/facebookresearch/visdom
- 支持数值（折线图，直方图等）、图像、文本以及视频等
- 支持Pytorch、Torch和Numpy
- 用户可以通过编程的方式组织可视化空间或者通过用户接口为数据打造仪表板，检查实验结果和调试代码
  - env:环境 & pane:窗格

### 安装

```bash
pip install visdom
```

启动服务

```bash
python -m visdom.server
```

## 40.tensorboardX介绍

各种不同的可视化展示方式

```bash
pip3 install tensorboardX
```

## 41.Torchvision介绍

- torchvision是独立于pytorch的关于图像操作的一些方便的工具库
  - https://github.com/pytorch/vision
  - https://pytorch.org/docs/master/torchvision/
- torchvision主要包括以下几个包：
  - vision.datasets: 几个常用的视觉数据集，可以下载和加载
  - vision.models: 已经训练好的模型，例如：AlexNet, VGG，ResNet
  - vision.transforms: 常用的图像操作，例如：随机切割，旋转，数据类型转换，图像到tensor，numpy数组到tensor，tensor到图像等
  - vision.utils、vision.io、vision.ops

------

## 42.机器学习与神经网络基本概念

人工智能领域--->机器学习--->神经网络（深度学习）

神经网络如下图所示：输入层、输出层、隐藏层、多层感知器

神经元、感知器、阶跃函数、激活函数

神经网络 vs 深度学习

- 多层感知器--->神经网络（简单的，3层）
- 多隐层的多层感知器--->深度学习（复杂的，隐藏层更多的神经网络）
  - CNN
  - RNN

前向运算forward

计算输出值的过程称为前向传播

conv层，pooling层，Relu层

反向传播

反向传播（BackPropagation, BP）神经网络训练方法

- 该算法于1986年由Rumelhar和Hinton等人提出，该方法解决了带隐藏层的神经网络优化的计算量问题，使得带隐藏层的神经网络走向真正的使用

反向传播算法，通过计算输出层结果与真实值之间的偏差来进行逐层调节参数

- 参数更新多少？---》导数和学习率

### 分类与回归

l1和l2

### 过拟合和欠拟合

- 过拟合：在训练集上表现好，但在测试集上效果差
- 欠拟合（高偏差）：模型拟合不够，在训练集上，表现效果差，没有充分的利用数据，预测的准确度低
- 偏差（Bias)：反映的是模型在样本上的输出与真实值之间的误差，即模型本身的精确度
- 方差（Variance）：

### 防止过拟合与欠拟合

防止过拟合方法

- 补充数据集（数据增强）
- 减少模型参数
- Dropout
- Earlystopping早停
- 正则化&稀疏化
- 等等

防止欠拟合方法

- 加大模型参数
- 减少正则化参数
- 更充分的训练
- 等等

### 正则化问题

L0， L1, L2, 无穷范数，核范数等等

- Pytorch通过weight_decay实现

Dropout

- nn.Dropout(p=0.5)

## 44.利用神经网络解决分类与回归问题（1）

### Pytorch搭建神经网络基本组成模块

- 数据
- 网络结构
- 损失
- 优化
- 测试
- 推理

### Pytorch完成波士顿房价预测模型搭建

波士顿房屋这些数据于1978年开始统计，共506个数据点，涵盖了波士顿不同郊区房屋14中特征的信息

https://t.cn/RfHTAgY

损失函数：MSE-LOSS

### Pytorch完成手写数字分类模型搭建

数据集为黑白图，大小为28x28

其中，包括60000张训练图片和10000张测试图片，

https://yann.lecun.com/exdb/mnist/

分类问题和回归问题，最本质的区别，就在于，一个是预测连续的值，一个是预测离散的值

分类问题采用Loss:交叉熵损失（信息论里的）

不确定性越大，熵越大

CrossEntropyLoss

### 模型的性能评价----交叉验证

简单交叉验证：训练集和测试集

2-折交叉验证

k-折交叉验证

留一交叉验证（数据缺失的情况下使用）

## 49.计算机视觉基本概念已处理

### 计算机视觉基础知识

人工智能与计算机视觉

人工智能是研究使计算机来模拟人的某些思维过程和智能行为（如学习，推理，思考，规划等等）的学科

计算机视觉基本任务：图像分类，图像检索，目标检测，图像分割图像生成，目标跟踪，超分辨率重构，关键点定位，图像降噪，多模态，图像加密，视频编解码，3D视觉。

### 颜色空间

颜色空间也称彩色模型，用于描述色彩

常见的颜色空间包括：RGB、CMYK、YUV、LAB、HSL和HSV/HSB

### RGB色彩模式

rgb色彩模式是工业界的一种颜色标准

通过对红、绿、蓝三个颜色通道的变化，以及它们相互之间的叠加，来得到各种各样的颜色的

红、绿、蓝三个颜色通道，每种色各分为256阶亮度

```
H x W x C
```

### HSV色彩模式

色相（Hue）、饱和度（Saturation）、明度（Value）

### 灰度图

每一个元素与图像的一个像素点相对应

### 常见的图像处理概念

亮度/对比度/饱和度

图像平滑与锐化

直方图均衡化

图像滤波

图像边缘拾取

### 图像平滑/降噪

图像平滑是指用于突出图像的宽大区域、低频成分、主干部分或抑制图像噪声和干扰高频成分的图像处理方法，使得图像亮度平缓渐变，减小突变梯度，改善图像质量。

- 归一化块滤波器
- 高斯滤波器
- 中值滤波器
- 双边滤波

图像锐化与图像平滑是相反的操作，锐化是通过增强高频分量来减少图像中的模糊，增强图像细节边缘和轮廓，增强灰度反差，便于后期对目标的识别和处理

锐化处理在增强图像边缘的同时也增加了图像的噪声

方法包括：微分法和高通滤波法

### 边缘提取算子

图像中的高频和低频的概念理解

通过微分的方式计算图像的边缘

- Roberts算子
- Prewitt算子
- sobel算子
- Canny算子
- Laplacian算子

### 直方图均衡化

直方图均衡化是将原图像通过某种变换，得到一幅灰度直方图为均匀分布的新图像的方法

对在图像中像素个数多的灰度级，进行展宽，而对像素个数少的灰度级进行缩减，从而达到清晰图像的目的

### 图像滤波

常见应用：去噪，图像增强，检测边缘，检测角点，模板匹配

- 均值滤波
- 中值滤波
- 高斯滤波
- 双边滤波

### 形态学运算

腐蚀，膨胀，开运算，闭运算，形态学梯度，顶帽，黑帽。

### OpenCV及其常用库函数介绍

```
imread
imshow
```

## 51.计算机视觉中的特征工程

特征工程就是一个把原始数据转变成特征的过程，这些特征可以很好的描述这些数据，并且里哟个它们建立的模型在未知数据上的表现性能可以达到最优（或者接近最佳性能）。从数学的角度来看，特征工程就是人工地去设计输入变量X

特征提取，特征选择，建模

卷积运算进行特征提取。

## 52. 卷积神经网络基本概念

卷积神经网络：以卷积层为主的深度网络神经

- 卷积层
- 激活层
- BN层
- 池化层
- FC层

对图像和滤波矩阵，做内积（逐个元素相乘再求和）的操作

### 常见的卷积操作

- 分组卷积（group参数）
- 空洞卷积（dilation参数）
- 深度可分离卷积（分组卷积+1x1卷积）
- 反卷积
- 可变形卷积

### 如何理解卷积层感受野

感受野，指的是神经网络中，神经元“看到的”输入区域，在卷积神经网络中，feature map上某个元素的计算受输入图像上某个区域的影响，这个区域即该元素的感受野。

### 如何理解卷积层的参数量与计算量

参数量：参与计算参数的个数，占用内存空间

FLOPs：浮点数运算数，理解为计算量，可以用来衡量算法/模型的复杂度

MAC：乘加次数，用来衡量计算量

### 如何压缩卷积层参数&计算量？

从感受野不变+减少参数量的角度，压缩卷积层

- 采用多个3x3卷积核代替大卷积核
- 采用深度可分离卷积
- 通道Shuffle
- Pooling层
- Stride=2

### 常见的卷积层组合结构

堆叠式-----UGG

跳连------ResNet

并联------inception

## 54.Pooling层

对输入的特征图进行压缩

- 一方面使特征图变小，简化网络计算复杂度
- 一方面进行特征压缩，提取主要特征

最大池化（Max Pooling），平均池化（Average Pooling）等

### 上采样层

Resize,如双线性插值直接缩放，类似于图像缩放，概念可见最邻近插值算法和双线性插值算法----图像缩放

反卷积，Deconvolution, 也叫Transposed Convolution

实现函数:

```python
nn.functional.interpolate()
```

```python
nn.ConvTranspose2d()
```

## 55.激活层

激活函数：为了增加网络的非线性，进而提升网络的表达能力

ReLU函数、Leakly ReLU函数、ELU函数等

```python
torch.nn.ReLu(inplace=True)
```

实际使用中，每一个卷积层后面都会配一个激活层，以增加网络的表达能力

### BatchNorm层

通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布，强行拉回到均值为0方差为0的标准正态分布

Batchnorm是归一化的一种手段，它会减小图像之间的绝对差异，突出相对差异，加快训练速度

不适用的问题：image-to-image以及对噪声敏感的任务

```python
nn.BatchNorm2d()
```

### 全连接层（FC层）

连接所有特征，将输出值送给分类器（如softmax分类器）

- 对前层的特征进行一个加权和，（卷积层是将数据输入映射到隐层特征空间），将特征空间通过线性变换映射到标准样本标记空间（也就是label）
- 可以通过1x1卷积+global average pooling代替
- 可以通过全连接层参数冗余
- 全连接层参数和尺寸相关

```python
nn.Linear()
```

可以理解为线性的模块，对图像尺寸敏感，参数多，需要配合dropout。可以理解为enbeding

### Dropout层

在不同的训练过程中，随机丢掉一部分神经元

测试过程中，不使用随机失活，所有的神经元都激活

为了防止或减轻过拟合而使用的函数，它一般用在全连接层

```
nn.dropout()
```

### 损失层

损失层：设置一个损失函数用来比较网络的输出和目标值，通过最小化损失，来驱动网络的训练

网络的损失通过前向操作计算，网络参数相对于损失函数的梯度，则通过反向操作计算

分类问题损失：(处理离散值)

```
nn.BCELoss()
nn.CrossEntropyLoss
```

回归问题损失：（处理连续值）

```
nn.L1Loss
nn.MSELoss
nn.SmoothL1Loss
```

## 56.经典卷积神经网络结构介绍

简单神经网络：LeNet,AlexNet,VGGNet

复杂神经网络：ResNet,InceptionNetV1 - V4, DenseNet

轻量型神经网络：MobileNetV1 - V3, ShuffleNet, squeezeNet

### 串联结构的典型代表

VGGNet

- 由牛津大学计算机视觉组和Google Deepmind共同设计
- 为了研究网络深度对模型准确度的影响，并采用小卷积堆叠的方式，来搭建整个网络结构

### 跳连结构的典型代表

ResNet

- 在2015年，由何凯明团队提出，引入跳连结构来防止梯度消失的问题，进而可以进一步加大网络深度
- 扩展结构：ResNetXt、DenseNet、Inception-ResNet

### 并行结构的典型代表

InceptionV1-V4(GoogleNet):

在设计网络结构时，不仅强调网络的深度，也会考虑网络的宽度，并将这种结构定义为Inception结构

## 57.轻量型结构的典型代表

MobileNetV1

由Google团队提出，并发表于CVPR-2017，主要目的是为了设计能够用于移动端的网络结构

使用Depth-wise Separable Convolution 的卷积方式，代替传统卷积方式，以达到减少网络权值参数的目的

MobileNetV2/V3

SqueezeNet/ShuffleNetV1/V2等等

## 58.多分枝网络结构的典型代表

多分枝结构

- SiameseNet
- TripleNet
- QuadrupletNet
- 多网络任务

相似性任务，人脸识别。样本的难例挖掘

## 59.Attention结构的典型代表

Attention机制

- 对于全局信息，注意力机制会重点关注一些特殊的目标区域，也就是所谓的注意力焦点，进而利用有限的注意力资源对信息进行筛选，提高信息处理的准确性和效率
- one-hot分布或者soft的软分布
- Soft---Attention或者Hard---Attention
- 可以作用在特征图上，尺度空间上，channel尺度上，不同时刻历史特征上

对不同的区域，进行加权

ResNet + Attention

SENet/Channel Attention

## 60.学习率(learn rate)

学习率作为监督学习以及深度学习中重要的超参，其决定着目标函数能都收敛到局部最小值以及何时收敛到最小值

合适的学习率能够使目标函数在合适的时间内收敛到局部最小值

```
torch.optim.Ir_scheduler
	ExponentialLR(指数)
	ReduceLROnPlateau（固定值）
	CyclicLR（周期）
```

## 61.优化器

对参数如何进行学习，如何进行调节

梯度下降算法。找到梯度下降最快的方向

GD、BGD、SGD、MBGD

- 引入了随机噪声

Momentum、NAG等

- 加入动量原则，具有加速梯度下降的作用

AdaGrad，RMSProp，Adam、AdaDelta

- 自适应学习率

```
torch.optim.Adam
```

## 62.卷积神经网络添加正则化

结构最小化，尽量采用更简单的模型

L1正则：参数绝对值之和

L2正则：参数的平方和（Pytorch自带，weight_decay）

直接定义在优化器参数中。

## 63.Pytorch实战计算机视觉任务-Cifar10图像分类

### 图像分类网络模型框架解读

分类网络的基本结构：

数据加载

数据预处理/增强

CNN网络

N维度向量

LOSS/Accuary

优化器

### 数据加载模块

RGB数据 or BGR数据

JPEG编码后的数据

torchvision.datasets中的数据集

torch.utils.data下的Dataset，DataLoader自定义数据集

### 数据增强

为什么需要数据增强?

数据增强的时候需要注意什么？

```
torchvision.transforms
```

### 网络结构

；

### 类别的概率分布

N维度向量对应N个类别

如何将卷积输出的tensor转换成N维度向量？FC，Conv，pooling都行

```
N x C x 1 x 1
```

SoftMax

### LOSS

标签转为向量，one-hot编码

```
nn.CrossEntropyLoss
```

label smoothing

### 分类问题常用评价指标

真实值，预测值

正确率，错误率，灵敏度，特效度，精度，召回率。

PR曲线、ROC曲线、AUV面积

如何绘制这个曲线？

去定义不同阈值

### 优化器选择

推荐使用：torch.optim.Adam

学习率初始值： lr=0.001

学习率指数衰减：

```
torch.optim.Ir_scheduler.ExponentialLR
```

## 65.Pytorch 编程实例之Cifar10-图像分类

### Cifar10/100数据集介绍&下载

从8000万个微小图像数据集的子集

http://www.cs.toronto.edu/~kriz/cifar.html

选择下载：

```
CIFAR-10 python version
```

直接下载比较慢，需要梯子

官网下提供了数据读取的脚本，直接粘进代码里就行了

## 78.分类问题优化思路

调参技巧：

- backbone		# 主干网络的选择，权衡精度和计算量
- 过拟合问题     # 解决方法：补充样本，数据增强，添加dropout层，添加L2正则项，模型改简单一些，使用最多的是添加数据增强
- 学习率调整     # 初始值选择，如何去衰减
- 优化函数        # 
- 数据增强        # 亮度、对比度、饱和度、裁剪、旋转
- https://www.imooc.com/article/305024

观察Loss ，观察数据，了解最新研究进展、网络结构， 分析错误case（需要具备图像处理的基本知识）

## 79. 分类问题最新研究进展和问题

研究领域：

- Backbone方面：
  - Google：EfficientNet, MobileNetV3    # 由此可知道，当前课程为2019年的
  - FaceBook AI: IdleBlock
- 小样本&0样本问题
  - few-shot
  - one-shot
  - zero-shot
- 数据增强方面
  - CutOut
  - MixUp
  - CutMix
  - AugMix
- 细粒度图像分类任务
  - 不仅要知道是鸟，还要知道其是哪种鸟
- 注意力机制

## 80.Pytorch实战计算机视觉任务之图像分割问题

### 概念介绍

图像分割：提取图像中，哪些像素是用来表述已知目标的

- 目标种类与数量问题
- 目标尺度问题
- 外在环境干扰问题
- 物体边缘

应用领域：

- 自然场景：Passcal VOC, COCO, ADE20K等
- 自动驾驶：Cityscapes, KITTI等
- 医学图像：DRIVE，IMT, IDRID等
- 航空遥感图像： inria Aerial Image Labeling Dataset等
- 感兴趣区域：MSRA Salient Object Databas等
- 场景文字：COCO_Text等

算法分类：语义分割，实例分割，全景分割

## 81.图像分割问题方法

### 语义分割

Encoding + Decoding 

为什么需要做下采样？

- 减小计算量
- 加大像素关系，空洞卷积

经典网络：

- UNet

- PSPNet
- SegNet
- Deeplab系列
- https://arxiv.org/pdf/1907.06199.pdf

上采样： conv+插值

### 实例分割

检测+分割

两阶段实例分割：

- Mask R-CNN， Mask Scoring R-CNN，PANet等

单阶段实例分割：

- acnchor-based:  YOLACT/YOLACT++
- acnchor-free: BlendMask, PolarMask, SOLOV1/SOLOV2

### 全景分割

实例分割+语义分割

Mask R-CNN（实例分割） + FPN（语义分割）

经典网络：

- Panoptic FPN (Mask R-CNN（实例分割） + FPN（语义分割）)
- AUNet
- UPSNet
- TASCNet
- JSIS-Net
- OANet

## 82.图像分割评价指标，及面临的挑战

算法性能评价

- Pixel Accuracy(像素准确率)
- MPA（平均像素准确率）
- Mean Intersection over Union(平均交并比)    
  - PA = (TP+TN) / (TP + TN + FP + TN)
  - IoU = TP / (TP + FP + FN)
- 频权交并比（FWIoU）
- FAIR研究团队为全景分割定了新的评价标准
  - PQ
  - SQ
  - RQ

### 图像分割问题中的难点

数据标注成本高

小样本问题

细粒度的图像分割任务

视频图像分割的边缘准确性和稳定性

模型压缩

全景分割：things类别和stuff类别

## 83. COCO图像分割数据集介绍

COCO数据集

- 微软发布的图像分类、对象检测、实例分割、图像语义的大规模数据集
- 图像包括91类目标，328000影像和2500000个label。其中语义分割的最大数据集，提供的类别有80类，有超过33万张图片，其中20万张有标注，整个数据集中个体的数目超过150万个
- https://cocodataset.org/#download
- 其他数据集
  - passcal VOC、SiftFlow、Stanford background, Vistas
  - Cityscapes、DAVIS、MINC、KITTI、ADE20K等

COCO数据格式介绍：

- json格式存放标注信息
  - “info”: 存储数据集的一些基本信息
  - “licenses”: 存储license信息
  - "categories": 存储数据集的类别信息，包括类别的超类、类别id、类别名称；
  - “images”: 存储这张图片的基本信息，包括图片名、长、宽、id等重要信息
  - “annotations": 存储这张图片的标注信息

## 84.Detectron框架介绍

2018年初，Facebook AI研究院（FAIR）公开了一个目标（视觉）检测平台，名叫Detectron（目前已经是V2）。它是一个软件系统，由Python语言和Caffe2深度学习框架构建而成

包括了：Mask RCNN, RetinaNet, Faster RCNN, RPN, Keypopints, Panoptic Segmentation等一系列优秀算法框架

支持数据集：COCO， passcal VOC， Cityscapes， LVIS等不同数据集

https://github.com/facebookresearch/detectron2

### 为什么会选择Detectron/MM-detectron这些框架？

快速验证现代主流深度学习算法

不仅仅模块化搭建网络，而且是模块化优化算法框架

节省寻找和测试深度学习模型效果的时间

我们需要更多的时间，跟进最新的研究进展

### 框架结构介绍

detectron2_master

- configs: 存储各种网络的yaml配置文件
- datasets: 存放数据集的地方
- detectron2
  - 运行代码的核心组件
  - config
    - compa.py: 对应之前的Detectron库
    - config.py: 
      - 定义DfgNode类
      - 提供get_cfg()方法，会返回一个含有默认配置的CfgNode，该默认配置值在default.py中定义。
    - default.py: 默认配置值
- tools：提供运行代码的入口以及一切可视化的代码文件
- projects： 提供真实项目代码示例

使用yacs来定义配置文件内容

https://github.com/facebookresearch/detectron2/blob/master/configs/Base-RCNN-FPN.yaml

环境配置：

https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

遇到问题这个文档里也有解决办法

训练&测试

- ```bash
  ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
  ```

  单显卡训练命令

- ```bash
  ./train_net.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml --eval-only MODEL>WEIGHTS /path/to/checkpoint_file
  ```

  测试命令

- ```
  ./train_net.py -h
  ```

  查看全部参数命令

案例

- ```bash
  python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
  ```

  

## 86.detectron2源码解读和模型训练-demo测试

### configs

进入：

```bash
cd detectron2/configs/COCO-InstanceSegmentation
```

vim查看：

```bash
vim mask_rcnn_R_50_FPN_1x.yaml
```

可以看到在上一级的文件中配置：

```
../Base-RCNN-FPN.yaml
```

退出并回到上一级，打开该配置文件：

```bash
vim Base-RCNN-FPN.yaml
```

可以看到训练网络的详细配置参数

根据github上的帮助文档，把下载好coco数据集，按照指定格式放置

后续将图片的路径补全

查看工具文件夹里的train_net.py

- set_up(): 加载配置文件

- Trainer(): 构建训练器，对模型进行初始化

- 训练

- 通过钩子函数打印log变化

进入配置文件夹：

```bash
cd detectron2/detectron2/configs/
```

打开defaults.py,查看网络训练配置参数

### backbone

进入模型文件夹：

```bash
cd detectron2/detectron2/modeling
```

做模型优化的时候，经常要去调整的地方

主干网络backbone

```bash
cd detectron2/detectron2/modeling/backbone
```

打开resnet.py

一般，如果主干网络太大，训练太慢，需要在这个脚本里，把batch_size改小

可以把resnet50改为resnet18

修改in_channel和out_channel还有feature

### data

进入data文件夹，这主要负责数据加载的脚本

```bash
cd detectron2/detectron2/data/datasets
```

查看coco.py

```python
load_coco_json()
coco_api.loadings(img_ids)
```

查看transform里数据增强的脚本文件transform.py

### demo

进入demo文件夹，把使用已有的模型推理

```
cd detectron2/demo
```

执行demo.py

```bash
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input1.jpg input2.jpg \
[--other-options]
--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

其中input后，要改为自己图片的路径，`[--other-options]`也要删掉

比如：

```bash
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input /mnt/e/PyProject/PytorchLearning/COCO/coco/val2017/000000000872.jpg \
--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

后来发现，没有所谓的vision模块，而那个py文件就在跟demo.py下面，修改导入语句，可以直接使用该文件里的函数

```python
from predictor import VisualizationDemo
```

### tools

自己训练模型

进入到tools里

```bash
cd detectron2/tools
```

```bash
./train_net.py -h
```

看到：

```bash
--config-file FILE    path to config file                                                                            
--resume              Whether to attempt to resume from the checkpoint directory. See documentation of                
                        `DefaultTrainer.resume_or_load()` for what it means.                                            
--eval-only           perform evaluation only 
```

需要定义的参数：

- ```python
  --config-file 	# 网络的配置
  ```

- ```python
  --resume		# 是否恢复训练
  ```

- ```python
  --eval-only		# 训练还是eval
  ```

首先，需要导入数据集

```bash
export DETECTRON2_DATASETS=/path/to/datasets
```

这里我们本地是：

```bash
export DETECTRON2_DATASETS=/mnt/e/PyProject/PytorchLearning/COCO/
```

输入训练命令：

```bash
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
```

基本上，要收敛的话，要训练一晚上

输出的结果，在tools/output文件夹里，而且也有log文件，可以同tensorboardX进行查看

```bash
tensorboard --logdir ./
```

不过好像要在训练中，才能查看，训练完了就不能看了

这里就不等它训练结束了，直接ctrl+C或+Z退出训练

### eval

采用eval对训练好的模型进行测试

```bash
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

这里的`/path/to/checkpoint_file`要替换成自己本地的path文件

```
./train_net.py \
  --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

不过，由于电脑不好，还没有训练出一个path文件，就停止了，导致无法eval

## 87.GAN的基础概念和典型模型介绍

GAN （Generative Adversarial Networks）是一种生成式的，对抗网络

- 生成网络，尽可能生成逼近样本，判别网络（Discriminator）则尽可能去判别该样本是真实样本，还是生成的假样本
- 生成器与辨别器，二者进行博弈对抗，直到各自胜率达到50%的平衡状态
- 然后，去掉辨别器，生成器就能生成以假乱真的样本

### GAN网络的优化目标函数

KL散度，JS散度，Wasserstein距离

### 常见的生成式模型

生成模型：给定训练数据，从相同的数据分布中生成新的样本

- PixelRNN/CNN：逐像素生成/从图像拐角处生成整个图像
- 自动编码器（AE）：编解码重构输入数据
- 变分自动编码器（VAE）：向自编码器加入随机银子获得的一种模型
- 生成对抗网络（GAN）：采用博弈论的方法

### 常见的GAN网络

GAN网络演进图谱

- GAN（2014）
  - WGAN（2017）
  - DCGAN（2015）:加入CNN卷积层
    - CycleGAN（2017）：基于DCGAN，将单项GAN网络，变成双向GAN网络，提出Cycle-Consistence Loss解决图片转换问题
      - StyleGAN(2019) :基于CycleGAN，加入风格迁移元素

### DCGAN

判别器和生成器都使用了卷积神经网络来代替GAN网络中的多层感知机，同时为了使整个网络可微，拿掉了CNN中的池化层，另外将全连接层以全局池化层替代以减轻计算量，加入BN使用激活函数RELU和LeakyReLU。

###  Pixel2pixel

Pixel2pixel，conditional GAN

- L1和L2 distance的loss，经常blur的模糊的image（需损失高频信息）
- UNet， l1 loss
- Markovian Discriminator (PatchGAN)

### CycleGAN

本质上，是两个镜像对称的GAN，构成了一个环形网络

两个GAN共享两个生成器，并各自带一个判别器，即共有两个判别器和两个生成器，一个单向GAN两个loss,两个即共四个loss

### StyleGAN

移除了传统的输入

映射网络

样式模块

随机变换

StyleGAN在面部生成任务中创造了新记录

### BigGAN

加大Batchsize

发现增加网络宽度可以提高IS分数，但加深网络的深度并没有明显的提升网络的效果

Truncation trick截断技巧

Orthogonal Regularization正交正则化

不稳定性的分析

8块GPU都要训练很久

GAN的优缺点

优点：

- GAN是一种生成式模型，相比较其他生成模型，波尔兹曼机和GSNs只用到了反向传播，而不需要复杂的马尔可夫链
- 相比于其他所有模型，GAN可以产生更加清晰真实的样本
- 无监督的学习方式训练，可以被广泛用在无监督学习和半监督学习领域
- GANs是渐进一致的，但VAE是有偏差的
- GAN应用到一些场景上，比如图片风格迁移，超分辨率，图像补全，去噪，避免了损失函数设计的困难

缺点：

- 训练GAN需要达到纳什均衡，有时候用梯度下降法做到，有时候做不到
- GAN不适合处理离散形式的数据，比如文本
- GAN存在训练不稳定，梯度消失，模式崩溃的问题

### 如何训练GAN网络

输入规范化到（-1， 1）之间，最后一层的激活函数使用tanh(BEGAN除外)

使用wassertein GAN的损失函数

如果有标签数据的话，尽量使用标签，也有人提出使用反转标签效果更好，另外使用标签平滑，单边标签平滑或者双边标签平滑

使用mini-batch norm，如果不用batch norm 可以使用instance norm 或者weight norm

避免使用RELU和pooling层，减少稀疏梯度的可能性，可以使用leakrelu激活函数

优化器尽量选择ADAM，学习率不要设置太大，初始1e-4可以参考，另外可以随着训练进行不断缩小学习率

给D的网络层增加高斯噪声，相当于是一种正则

### GAN网络有哪些重要应用？

图像数据生成

超分辨率重构

音乐生成

图像转换/翻译

图像合成

场景合成

人脸合成

文本到图像的合成

风格迁移

图像域的转换

图像修复

MashGAN

去雨/去雾

年龄仿真

## 89.图像风格转换数据下载与自定义dataset类

可以直接看cyclegan在github上的源码，里面提供了数据集，并且可以搭建gan网络

https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$FILE.zip

先写好脚本，然后

进入到本地目录中去

```
cd /mnt/e/PyProject/PytorchLearning/cycleGAN
```

执行以下脚本，下载数据集：

```
python download_cyclegan_dataset.py apple2orange
```



## 94.RNN网络基础

### 基础概念

主要是解决序列建模的任务。

- 比如说，语音数据，
- 比如说，不同时间段内，采集到的和时间相关的一些数据

RNN网络的记忆性

- 可以利用一些历史信息，对将来的数据进行预测

可以利用任意长序列的信息（理论上）

- 实际上有个长度范围

存在“梯度消失”问题，在实际中，只回溯利用与他接近的time steps上的信息

### RNN vs CNN

RNN的假设：事物的发展是按照时间序列展开的，即前一刻发生的事物会对将来的抒事情的发展产生影响

CNN的假设：人类的视觉总是会关注实现内特征最明显的点，而CNN神经网络，是模仿人类处理信息的过程

RNN具有记忆能力:上一时刻隐藏层的状态，参与到了这个时刻的计算过程中，

RNN主要用于序列问题建模

### RNN网络应用场景

语音识别、OCR识别（图片转化为文本的过程）、文本分类、序列标注、音乐发生器、情感分类、DNA序列分析、机器翻译、视频动作识别、命名实体识别

## 95.常见的RNN结构---simple RNN网络

常见的RNN有：

simple RNN ==》Bi-RNN ==》 LSTM BI-LSTM

Seq2Seq / Attention ==》Transformer ==》BERT

### simple RNN原理

simple RNN是后面这些RNN结构的的基本单元，

举个例子，在一个订票系统上，我们输入“Arrive Beijing on November 2nd” 这样一个序列，希望算法能够将关键词‘beijing'放入目的地

再次输入“Leave Beijing on November 2nd” 希望将’beijing‘放在出发地。

循环神经网络，就是根据第一次记录到的目的地，自动的推算出第二次的出发地。

我们希望能够让神经网络拥有”记忆“的能力，能够根据之前的信息（这个例子中是Arrive或Leave）从而得到不同的输出

### 前向运算

### 反向运算

RNN网络长序列学习，非常容易出现梯度消失或者梯度爆炸的现象

### 不同结构

表达了，不同的，从序列到序列的关系

一对一，一对多，多对一，多对多

### 存在的问题

梯度消失，梯度爆炸，constant error carrousel(CEC)（长序列问题）

## 96. Bidirectional RNN原理介绍（双向RNN）

### 基本原理

假设当前的输出不仅仅和之前的序列有关，并且还与之后的序列有关

两个simple RNN叠加在一起

拥有两个隐藏层，一个正向，一个反向

### 单层双向

### 多层双向

## 97.LSTM原理介绍

LSTM是为了避免长依赖问题而精心设计的

- 记住较长的历史信息，实际上是它们的默认行为，而不是他们呢努力学习的东西
- 一种特殊的RNN模型
- 为了解决RNN模型梯度弥散的问题而提出的

在传统的RNN中，训练算法使用的是BPTT，当时间比较长时，需要回传的残差会指数下降，导致网络权重更新缓慢，无法体现出RNN的长期记忆的效果，因此需要一个存储单元来存储记忆

在标准的RNN中，该重复模块将具有非常简单的结构，例如：单个tanh层

LSTM不同于单一神经网络，这里是由四个，以一种非常特殊的方式，进行交互

LSTM的关键就是细胞状态，水平线在图上方贯穿运行

- 细胞状态类似于传送带，直接在整个链上运行，只有一些少量的线性交互，信息在上面流传保持不变会很容易

门结构（LSTM拥有三个门，来保护和控制细胞状态）

- 去除或者增加信息到细胞状态的能力
- 门是一种让信息选择式通过的方法，sigmoid + pointwise(点乘)

忘记门：决定我们会从细胞状态中丢弃什么信息

输入层门：确定什么样的新信息被存放在细胞状态中。Sigmoid决定需要被更新的值、tanh层创建一个新的候选值向量加入状态中。确定更新的信息

输出层门：我们需要确定输出什么值，这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本

### Bi-LSTM原理介绍

### LSTM网络结构变种

Gated Recurrent Unit(GRU) （使用最多的）

Clockwork RNN

Grid LSTM

生成模型的RNN

## 98.序列任务中的Attention机制

通常使用传统编码器-解码器的RNN模型

- 先用一些LSTM单元来对输入序列进行学习，编码为固定长度的向量表示
- 再使用一些LSTM单元来读取这种向量并解码为输出序列

对于序列到序列的任务，存在两个非常致命的问题就是：

- 输入序列不论长短都会被编码成一个固定长度的向量表示，而由于不同的时间片或者空间位置的信息量明显有差别，利用定长表示则会带来误差的损失
- 当输入序列比较长是，模型的性能会变得很差

Attention机制通过对输入信息进行选择性的学习，来建立序列之间的关联

这非常适合序列到序列的学习任务，比如：机器翻译，自动问答，语音识别等

seq2seq + Attention模型

### Seq2Seq模型

从输入到输出，存在多种不同的RNN结构

Seq2Seq实际上就是many to many模型的一种（encoder-decoder结构）

从一个序列到另外一个序列的转换

翻译功能、聊天机器人对话模型等不同场景

encoder-decoder结构先将输入数据编码成一个上下文向量c

可以把Encoder的最后一个隐状态赋值给c，还可以对最后的隐状态做一个变换，得到c，也可以对所有的隐状态做变换

另一个RNN网络对c进行解码，这部分RNN网络被称为Decoder

将c当作每一步的输入

在每个时间输入不同的c来解决这个问题

### seq2seq + Attention模型的应用领域

机器翻译

文本摘要

阅读理解

语音识别

OCR识别

## 99.Transformer

Google提出《Attention is all you need》

用全attention的结构代替了lstm

- 两个sub-layer组成
- multi-head self-attention mechanism
- fully connected feed-forward network
- residual connection&normalisation

attention机制

scaled dot-Product attention(缩放点积)

self-attention机制

- Q V K相等的结构
- 全部都用query来进行初始化

multi-head self-attention

- 将h个不同的attention结果拼接起来

Feed-forward networks

- 位置全链接前馈网络----MLP变形
- 增加非线性表达能力

masked multi-head attention

- 对于decoder中的第一个多头注意力子层，需要添加masking，确保预测位置i的时候，仅仅依赖于位置小于i的输出，确保预测第i个位置时，不会接触到未来的信息

Positional Encoding

- 引入序列信息（顺序）
- 使用sin编码和cos编码的原因，是可以得到词语之间的相对位置

### 优点

计算复杂度降低

并行计算

计算一个序列长度为n的信息，要经过的路径长度只需要一步矩阵计算

- cnn需要增加卷积层数来扩大视野
- rnn需要从1到n逐个进行计算，所以也可以看出，self-attention可以比rnn更好的解决长时依赖问题

### 缺点

实践上：有些rnn轻易可以解决的问题，transformer没做到，比如复制string，或者推理时碰到的sequence长度比训练时更长（因为碰到了没见过的position embedding）

理论上，transformers非computationally universal(图灵完备)

参考：《universal transformers》

## 100.BERT

语言模型：通过在海量的语料的基础上，运行自监督学习方法，为单词学习一个好的特征表示

- 使用了Transformer作为算法的主要框架，能更彻底的捕捉语句中的双向关系
- 使用了mask language model（MLM）和next sentence prediction（NSP）的多任务训练目标
- 使用更加强大的机器训练更大规模的数据，使BERT的结果达到了全新的高度
- Google开源了BRERT模型，用户可以直接使用BERT作为Word2Vec的转换矩阵，并高效的将其应用到自己的任务中

### mask language model（MLM）

将单词序列输入给BERT之前，每个序列中有15%的单词被[MASK]token替换，然后，模型尝试基于序列中其他未被mask的单词的上下文来预测被掩盖的原单词

### next sentence prediction（NSP）

模型接收成对的句子作为输入，并且预测其中第二个句子是否在原始文档中也是后续的句子

在训练期间，50%的输入对在原始文档中是前后关系，另外50%中是从语料库中随机组成的，并且是于第一句断开的。
