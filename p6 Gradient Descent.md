## Gradient Descent

![1639367826857](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639367826857.png)

​	神经网络与一般的线性模型最大的差别在于神经网络有很多很多的参数

### Chain Rule（链式求导法则）

![1639368047194](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639368047194.png)

​		输出结果与真实值的差距$c^n$ 累加就成为总的损失值，后面对这个总损失值求偏导

![1639368225407](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639368225407.png)

反向传播——正向推导

![1639368700759](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639368700759.png)

反向传播——逆向推导

![1639368815646](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639368815646.png)

![1639376244290](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639376244290.png)

![1639376774323](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\1639376774323.png)

## 类神经网络训练不起来怎么办——（一）局部最小值和鞍点

通常我们在做优化的时候，发现**随着你的参数不断update，你的training的loss不会下降**，这个时候需要比较你的模型，如果说你发现你采用的模型没有发挥应有的效果，那么此时优化阶段显然是有问题的。

### loss搞不动

但有时你会发现，你的模型一开始就训练不起来，那么这时候不管你怎么调参，损失函数都掉不下去，这个时候可能发生的事情：

![image-20210314153200619](https://camo.githubusercontent.com/230f25bc4576853b47c1f3774549449b5fc3622143c8b188659ef9876559f3fd/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343135333230303631392e706e67)

这个时候你会发现，你可能是落在局部最小值处，也可能是落在鞍点处。

那么我们如何知道到底是哪种情况呢？

![image-20210314153005913](https://camo.githubusercontent.com/9411a3bd1f98936dfa9e1c0c4e629a22f149199221ce03138041d14f6b79a86f/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343135333030353931332e706e67)

### 相关数学求解

这里涉及微积分和线性代数的知识：

![image-20210314154450970](https://camo.githubusercontent.com/4c30d580b2a6d6202eac302e288b638ddb0217d5ac91d4e0ddc0bbff6606d212/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343135343435303937302e706e67)

- 第一项是$L(θ')$,就告诉我们说,当$θ$跟$θ'$很近的时候,$L(θ)$应该跟$L(θ')$还蛮靠近的
- 第二项是$(θ-θ')^Tg$

![image-20210314155508574](https://camo.githubusercontent.com/950e3130053a439dcfdedec243ddf38db056f864b9c42fc78732a203bc8e129e/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343135353530383537342e706e67)

​	第三项跟Hessian有关,这边有一个$H $

![image-20210314155802228](https://camo.githubusercontent.com/27e27bc5c3e0973730697c504aa0e3dc2127c55e7bb0c1ca8c6783b3b7858280/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343135353830323232382e706e67)

总结一下：**gradient就是一次微分,hessian就是裡面有二次微分的项目**

### Hessian对可疑点进行判断

如果我们走到一个可疑点，意味着gradient为0，绿色的一项可以直接划掉

![image-20210314160538203](https://camo.githubusercontent.com/8594c8a521f35b88f39e6e92c39b0468d90b2f8aee38c855cf6f1cdeb72ed2a1/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343136303533383230332e706e67)  

  **我们就可以根据红色项判断error surface的情况**

![image-20210314161411744](https://camo.githubusercontent.com/ab2823c690f6f9692930e61e55de965a3d7bb3f82ad80650b876c3a0a7c50ae0/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343136313431313734342e706e67)

对所有的v而言,$v^THv$都大於零,那这种矩阵叫做**positive definite 正定矩阵**,positive definite的矩阵,**它所有的eigen value特征值都是正的**

如果你今天算出一个hessian，我们就可以直接去看H矩阵的特征值eigen value

- **所有eigen value都是正的**,那就代表说这个条件成立,就$v^THv$,会大於零,也就代表说是一个local minima。所以你从hessian metric可以看出,它是不是local minima,你只要算出hessian metric算完以后,看它的eigen value发现都是正的,它就是local minima。
- 那反过来说也是一样,如果今天在这个状况,对所有的v而言,$v^THv$小於零,那H是negative definite,那就代表所有**eigen value都是负的**,就保证他是local maxima
- **那如果eigen value有正有负**,那就代表是saddle point,

### 举例说明

![image-20210314192209655](https://camo.githubusercontent.com/25590ed312c39354948fefbe2c8e9e422f4a97678e6255ad426d818b3a607d01/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343139323230393635352e706e67)

### 不用害怕Saddle Point

Saddle Point不但能够指出参数，还能指出更新的方向

![image-20210314200048825](https://camo.githubusercontent.com/8043617efd466010ffe09733599081df871f280c4763d7784c6032b9c184f3e0/68747470733a2f2f67697465652e636f6d2f756e636c657374726f6e672f646565702d6c6561726e696e6732315f6e6f74652f7261772f6d61737465722f696d676265642f696d6167652d32303231303331343230303034383832352e706e67)

我们$\begin{matrix}1\\1\end{matrix}$取是它对应的一个eigen vector,那我们其实只要顺著这个u的方向,顺著$\begin{matrix}1\\1\end{matrix}$

 这个vector的方向,去更新我们的参数,就可以找到一个,比saddle point的loss还要更低的点

当然，实际的作业中，几乎不会把海赛矩阵算出来。

### 鞍点vs局部最小值

**从三维空间来看，没有路可以走了；拓展到高维是否有路可走呢**

![image-20210314205016598](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210314205016598.png)

我们就有了思路，如果我们增加参数，会不会减少local minima的个数呢？

如果你自己做实验，也支持这个假说：

![image-20210314205517453](https://gitee.com/unclestrong/deep-learning21_note/raw/master/imgbed/image-20210314205517453.png)

这张图的意思：越往右代表我们的critical point越像local minima，但是没有真正变成local minima，即使在最极端的情况下依然有一半的样例可以让loss下降

所以，在经验上看，local minima并没有那么常见多数时候，你发现你的梯度真的非常小，参数不再update了，往往是因为卡在了saddle point