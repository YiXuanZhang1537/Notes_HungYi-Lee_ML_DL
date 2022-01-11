# P3 回归 Regression

## 概念：

Regression就是找到一个函数function，输入特征x进而输出数值scalar

## 应用举例：

股市预测、自动驾驶、商品推荐

## 模型步骤：

- step1：模型假设，选择模型框架（线性模型）
- step2：模型评估，如何判断众多模型的好坏（损失函数）
- step3：模型优化，如何筛选最优的模型（梯度下降）

### Step1：模型假设，选择模型框架（线性模型）

**一元线性模型（单个特征）**

用一个特征代入线性模型进行计算

**多元线性特征（多个特征）**

选不同的性质，构成w矩阵，代入计算

y=b+∑wixi

### Step2：模型评估-损失函数

#### 收集和查看训练数据

https://github.com/YiXuanZhang1537/Notes_HungYi-Lee_ML_DL-2017/blob/main/image/chapter3-3.png



将10组原始数据在二维图中展示，图中的每一个点 (x_{cp}^n,\hat{y}^n)(xcpn,y^n) 对应着 进化前的CP值 和 进化后的CP值。

#### 判断众多模型的好坏

定义损失函数

https://github.com/YiXuanZhang1537/Notes_HungYi-Lee_ML_DL-2017/blob/main/image/chapter3-4.png

将w,b在二维坐标图上展示

https://github.com/YiXuanZhang1537/Notes_HungYi-Lee_ML_DL-2017/blob/main/image/chapter3-5.png

（颜色越深代表模型更优）

### Step3:最佳模型-梯度下降

需要引入一个概念——学习率：移动的步长

- 步骤1：随机选取一个 w^0w0
- 步骤2：计算微分，也就是当前的斜率，根据斜率来判定移动的方向
  - 大于0向右移动（增加ww）
  - 小于0向左移动（减少ww）
- 步骤3：根据学习率移动
- 重复步骤2和步骤3，直到找到最低点

每一步“迈出”的步伐取决于两方面——学习率以及当前位置的斜率

https://github.com/YiXuanZhang1537/Notes_HungYi-Lee_ML_DL-2017/blob/main/image/chapter3-9.png

经过多次迭代，会找到local optimal

但是会有一个问题，不是global optimal（整体最优解）

如果是两个参数，就分别执行随机初始化、更新参数两个操作

如果用图形化的手段表示出来，是这样子的：

https://github.com/YiXuanZhang1537/Notes_HungYi-Lee_ML_DL-2017/blob/main/image/chapter3-11.png

#### 梯度下降算法在现实世界遇到的挑战

- 问题1：当前最优（Stuck at local minima）
- 问题2：等于0（Stuck at saddle point）
- 问题3：趋近于0（Very slow at the plateau）

其实在线性模型里面都是一个碗的形状（山谷形状），梯度下降基本上都能找到最优点，但是再其他更复杂的模型里面，就会遇到 问题2 和 问题3 了

#### w和b偏微分的计算方法

https://github.com/YiXuanZhang1537/Notes_HungYi-Lee_ML_DL-2017/blob/main/image/chapter3-14.png

### 验证训练好的模型的好坏

平均误差评价模型的好坏

#### 更强大复杂的模型：1元N次线性模型

引入高阶，达到多次拟合的效果

#### 过拟合问题

在训练集上表现为更优秀，到达测试集上反而变差的情况

#### 步骤优化

Step1优化，2个input的四个线性模型合并到一个线性模型中

Step2优化，希望模型更强大表现更好（更多参数，更多input）

将血量（HP）、重量（Weight）、高度（Height）也加入到模型中

Step3优化，加入正则化

更多特征，但是权重 w 可能会使某些特征权值过高，仍旧导致overfitting，所以加入正则化
