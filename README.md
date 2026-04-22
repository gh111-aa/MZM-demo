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




### 定义模型与训练脚本


























