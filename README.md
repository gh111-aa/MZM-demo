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

























