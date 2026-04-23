# MZM-demo
***
这是一个初学者对神经网络和逆向设计流程的体验
***
## 项目架构
```plaintext
mzm_forward_model/
 │
 ├── data/                         # 数据存放与预处理
 │   ├── raw/                      # 原始数据集（只读）
 │   │   └── Sim_MZM_dataset.txt
 │   ├── processed/                # 处理后的训练/验证/测试集（.npy/.h5）
 │   │   ├── X_train.npy
 │   │   ├── y_train.npy
 │   │   ├── X_val.npy
 │   │   ├── y_val.npy
 │   │   ├── X_test.npy
 │   │   ├── y_test.npy
 │   │   └── scalers.pkl           # 保存归一化参数（输入/输出各一个）
 │   └── prepare_data.py           # 数据预处理脚本（划分、归一化）
 │
 ├── models/                       # 模型定义与训练组件
 │   ├── __init__.py
 │   ├── mlp_model.py              # 定义带残差、BN、Dropout的MLP类
 │   ├── physics_loss.py           # 计算物理约束项（偏导数惩罚）
 │   └── train.py                  # 主训练脚本（包含优化器、调度器、早停）
 │
 ├── configs/                      # 配置文件（超参数）
 │   └── config.yaml               # 网络结构、训练参数、路径等
 │
 ├── scripts/                      # 辅助工具脚本
 │   ├── inference.py              # 单次/批量推理示例
 │   └── evaluate.py               # 测试集评估、绘图
 │
 ├── logs/                         # 训练日志（TensorBoard）
 │   └── run_YYYYMMDD_HHMMSS/
 │
 ├── checkpoints/                  # 模型保存目录
 │   └── best_model.pth
 │
 ├── notebooks/                    # Jupyter探索性分析（可选）
 │   └── 01_data_exploration.ipynb
 │
 ├── requirements.txt              # Python依赖
 ├── README.md                     # 项目说明与使用指南
 └── .gitignore
```

## 首先我们搭建环境和依赖包

这里搭建了pytorch并且验证了所需要的依赖：
1. python --version #验证python的版本
2. 验证pytorch的版本
3. 验证所需要的包
<img width="864" height="290" alt="d186982d00c2b9cc65bf92508be351ad" src="https://github.com/user-attachments/assets/78fd2a07-3040-444f-ba1f-08607b8241e0" />

## 任务的核心实施阶段
### 创建项目目录结构  
### 编写data/prepare_data.py脚本  
这个文件的目的是为了对raw里面的初始数据进行数据预处理，并且将处理之后的结果文件放入data/processed文件夹里面  
这些结果文件包括：
<img width="1041" height="599" alt="c1421832b52563ac536d95871bb26de9" src="https://github.com/user-attachments/assets/74ffd215-59f6-4296-b2a4-3d2346b758ea" />
***
> **知识点:** `epoch`、`batch`、`lteration`:  
`epoch`:表示所有数据集被循环一次  
`batch`:表示每次训练使用的一小批数据集的个数  
`literation`:表示训练完一轮batch的参数更新过程  
***
> **知识点:** `验证集`:  
`验证集(validation)`:验证集的主要作用是为了调整超参数和终止训练  
验证集是在每个epoch进行完之后使用的，但是同样的，验证集的信息也在这个时候流入了模型（进行了超参数调整或者模型终止），如果测试模型的时候也用验证集的话就会过拟合，所以测试模型的时候使用一个完全未知的数据集，也就是验证集（test）。
***
### 运行脚本生成处理后的数据  
这里对数据进行了一个清洗，保证V_pi值小于500，大于0  
### 定义模型与训练脚本  
此处篇幅不够，重开一个标题  

## 定义模型与训练脚本
这一部分需要三个py文件，分别是  
搭建网络结构文件：models/mlp_model.py  
模型超参数配置文件：configs/config.yaml  
开始训练模型的文件：models/train.py  

如果要引入pinn（物理约束）的话，还要有一个物理Loss文件：models/physics_loss.py  
或者将物理约束loss嵌入models/train.py文件中  

### 搭建网络结构mlp_model.py
注意：一个神经元的过程包括：  
1. 使用自己独特的权重对上一层神经元线性求和
2. 激活函数
3. 输出  

所以每个神经元的初始权重必须是不一样的，所以有随机初始化的方法。  
***
> **知识点：** `BatchNorm`：  
`BatchNorm`是针对每一个神经元的线性加权求和之后、激活函数之前的数值进行归一化的
要注意它的归一化对象是单个神经元面对的batch  
比如这一层中的某个固定神经元，要面对具有200个神经元的上个隐藏层，如果batch_size是30的话，那么这个神经元就面临30个200维的数据
首先这个神经元会直接利用权重向量对200维的加权成1维，也就是产生了三十个1维数据，然后使用batchnorm，以这三十个数据为总体，对这三十个数据进行归一化
从而再进行激活函数处理，输出这个神经元的最终结果。

> 使用`batchnorm`的好处：
1. 解决了内部协变量偏移的问题，让网络专注于训练，减少了由于数据分布不断变动所产生的影响。
2. 对于ReLU（你的项目中使用的是ReLU）：最广泛和最推荐的做法是将BatchNorm放置在ReLU激活函数之前。这样做，BatchNorm可以将数据归一化到以0为中心的分布，避免ReLU因输入为负而“杀死”过多的神经元。

> 这和学习率的关系：
加了`batchnorm`之后，mini_batch就被控制在分布01的正态分布上，数据的分布波动被控制，loss的变化也被控制，那么权重的迭代步伐被缩小
所以可以采取更大的学习率。  
以往是大步伐少步走，加了`batchnorm`之后是小步伐多次走，更容易接近最优权重。
***

### 撰写配置文件config.yaml  
yaml文件的好处就是可以直接将框架中的超参数写道另一个文件中直接调用，简单明了，把超参数从原本python文件中抽离出来，减少修改错误。























