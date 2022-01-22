## Convolutional Neural Network(CNN)

### 图像的分类

![1639312378260](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639312378260.png)



假设输入的照片尺寸一致，在丢进模型之前应该做的事情——Rescale

模型的目标是分类，所以应该把分类表示成一个==One-Hot的Factor（比如图片是一只猫，猫所对应的维度数值就是零）==

得到的$\hat y$ 就可以辨别处多少不同种类的东西，比如维度是2000，就可以辨识出2000种以上的东西

我们也希望经过Softmax的y’ 它和$\hat y$的交叉熵越少越好

重点是怎么把一张图像进行输入？

首先，我们要看下电脑是怎么看待这张图片的（三维的tensor，tensor是超过二维的矩阵）

![68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303332393135303435333037332e706e67](E:\CityU Data Science\九松NLP\68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303332393135303435333037332e706e67.png)

三个参数，一个是长，一个是高（长与宽代表解析度）

一个是channel（RGB三种颜色）

我们只要==把图片展成一个向量==，我们就可以进行输入

<u>100×100×3向量中的每一项就是某一个颜色的强度</u>（Pixel）

之后，我们把这个一列向量输入全连接的神经网络

![1639313241015](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639313241015.png)



这个时候会发现，一共有3×10^7个参数，==参数越多确实能够增加弹性，但是也容易出现过拟合的问题==（下周解释过拟合背后的数学原因）

那么怎么避免过拟合呢？考虑图像的特性，我们不一定采用全连接，我们可以让部分权重消失

### 观察1 识别某些可疑的图案

神经网络可以识别重要的特征图案，得出结论

![1639313632531](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639313632531.png)

如果我们要神经网络去捕捉重要特征，那么我们没有必要让神经网络去看完整的图片，只需要截取一小部分当作输入就足够判断了。

### 简化1 选择一个Receptive field

这样对应的函数就只负责这一部分

![1639314748441](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639314748441.png)

选择区域也可以重叠，形状完全可以自定义

![1639377910382](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639377910382.png)

### 最经典的Receptive field 

![1639378250214](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639378250214.png)

这里牵涉几个概念——<u>1.这个算法会观察所有的Channel</u> 

​									<u>2.它会有重叠，扫描的区域会移动,移动的范围叫stride</u>

​									<u>3.当出现padding（超出图像区域），会进行补齐（按照0处理）</u>

​									<u>4.高与宽合起来叫Kernel Size</u>

通过这种方式，图像的每个部分都会被侦测到

### 观察2 同样的图像出现在不同的位置

![image-20210329182937549](https://camo.githubusercontent.com/c23834b45716ddeabdf2202042dcceb7b6deb37783ebf9234ea54d5f4db6d58a/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303332393138323933373534392e706e67)

上下两张图干的都是一个事情，都是在侦测鸟嘴。 那么既然侦测的都是同一个事情，只是守备的范围不一样，那么我们需要每一个守备范围都去做鸟嘴检测吗？

### 简化2 能否让不同Reception Filed之间共享参数？

共享，是指两部分共享相同的权重

![1639387796282](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639387796282.png)

两个 Neuron 守备的范围不一样,就算它们的参数一样,它们的输出也不会是一样的

（为什么要加bias，可以使得神经网络Fit的范围得到左右的调整，参考资料——https://www.zhihu.com/question/68247574）

其实**每一个 Receptive Field都只有一组参数而已**，共用的参数称为field

![1639388185033](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639388185033.png)

### Summary

![1639388331472](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639388331472.png)



全连接层里面加入Receptive Field,并且参数共享，就形成了卷积层，卷积层组成的模型称为CNN

<u>（为什么CNN有比较大的Model bias）</u>

那么与全连接层相比，卷积层有大的偏差，但这并不一定是坏事

全连接针对的是全部的情景，对于具体情景适应能力较差；卷积层针对图像方面，在图像方面偏差大是可以解决的

### Another story based on filter

![1639388887798](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639388887798.png)

channel=3 彩色；channel=1 黑白

每一个filter的目标是去图片里面抓取pattern

### filter怎么抓取pattern

现在举一个实际的例子：

![1639467002164](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639467002164.png)

它是一个 6 × 6 的大小的图片,那这些 Filter的做法就是,先把 Filter 放在图片的左上角,然后**把 Filter 裡面所有的值,跟左上角这个范围内的 9 个值做相乘**

注意！在此处卷集运算中，老师讲的**内积**，**不要理解为**线性代数中**矩阵的乘法**，而是**filter**跟**图片对应位置**的数值**直接相乘**，所有的都乘完以后**再相加**

在下图中 filter为

$
 \left[
 \begin{matrix}
   1 &0 & 1 \\
   0 & 1 & 0 \\
   1 & 0 & 1
  \end{matrix}
  \right] \tag{3}
$

![img](https://easy-ai.oss-cn-shanghai.aliyuncs.com/2019-06-19-juanji.gif)

其中，移动的距离叫做stride

因此，每一个filter都会返回一组数字；64个filter就会返回64群数字

![image-20210330161535127](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330161535127.png)

**Filter 的这个高度就是它要处理的影像的 Channel**,所以跟刚才第一层的 Convolution,假设输入的影像是黑白的 Channel是 1,那我们的 Filter 的高度就是 1,输入的影像如果是彩色的 Channel 是 3,那 Filter 的高度就是 3,那在第二层裡面,我们也会得到一张影像,对第二个 Convolutional Layer 来说,它的输入也是一张图片,那这个图片的 Channel 是多少,这个图片的 Channel 是 64（64是前一个convolution的Filter数目）

### **如果我们的 Filter 的大小一直设 3 × 3,会不会无法看大范围的Pattern呢**

![image-20210330162532833](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330162532833.png)

我们其实可以从这张图看出，下面那张图的扫描范围其实对应的是上面矩阵的5*5，所以你的Neural Network足够深的时候是不必要担心漏掉扫描区域的

![image-20210330163033020](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330163033020.png)

其实在第一个版本里面共用参数那件事情就是第二个版本里面的Filter。而且在Filter其实也是有bias的

把Filter扫过一张图片其实就是Convolution，这也就是Convolution Layer的来由

![image-20210330164622894](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330164622894.png)

### CNN是基于两个观察

①我们不需要看整张图片,那对 Neuron 的故事版本,对於第一个故事而言就是,Neuron 只看图片的一小部分,对 Filter 的故事而言就是,我们有一组 Filter,每个 Filter 只看一个小范围,它只侦测小的 Pattern

②同样的 Pattern,可能出现在图片的不同的地方,所以 Neuron 间可以共用参数,对 Filter 的故事而言就是,一个 Filter 要扫过整张图片,这个就是 Convolutional Layer

### 卷积层第三个常用的东西——Pooling（池化）

![image-20210330164857135](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330164857135.png)

​	Pooling类似于ReLU等激活函数，是固定好的功能，不需要根据Data进行学习。

现在讲解的是Max Pooling怎么运作：

![1639468403494](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639468403494.png)

要做Pooling的时候，我们把这些数字分成几个几个一组，比如在上图的例子里面是2*2个一组，每一组里面选一个代表，我们选最大的那个（红色框）

![1639468525840](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639468525840.png)

### Convolutional Layers+Pooling

![image-20210330170700272](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330170700272.png)

Pooling主要是把图片变小==做完 Convolution 以后我们会得到一张图片,这一张图片裡面有很多的 Channel,那做完 Pooling 以后,我们就是把这张图片的 Channel 不变,本来 64 个 Channel 还是 64 个 Channel,但是我们会把图片变得比较狭长一点==

当然，Pooling对于图像也是有伤害的：假如你做的是非常微细的东西，用Pooling来处理表现会稍差一些

近年来出现很多把Pooling丢弃的事情，这是近年来运算能力越来越强的缘故

### 全部CNN的架构

![image-20210330171219295](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330171219295.png)

### CNN的应用——Playing Go

下棋子其实就是分类的问题，用CNN解决更好

![image-20210330172015868](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330172015868.png)

Pooling不一定要用，Pooling不一定是好的

![image-20210330183705782](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210330183705782.png)

### 更多应用：影像、语音

不能直接把图像处理方面的CNN直接应用在影像和语音方面，不然有可能不工作

### CNN的弱点

不能处理放大缩小，在使用CNN之前需要Data Augmentation

有一个架构可以处理这类问题——Spatial Transformer Layer