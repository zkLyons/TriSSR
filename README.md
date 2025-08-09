@author 张康
@time 2025-8-9

# TriSSR
我的第一个推荐系统模型
It is a novel generic and efficient sequential recommendation backbone, which explores the seamless adaptation of **Mamba** for sequential recommendations. Specifically, SSD4Rec marks the variable- and long-length item sequences with sequence registers and processes the item representations with bidirectional Structured State Space Duality (SSD) blocks. This not only allows for hardware-aware matrix multiplication but also empowers outstanding capabilities in **variable-length and long-range sequence modeling**.
这是一个有效且新颖的推荐系统模型，有效结合了状态空间模型、频域滤波模型和时间编码器进行特征提。并设计了三特征融合器，将状态空间信息、频率特征和时间特征无缝融合起来，实现了性能上的提升。最后模型在四个公开数据集上的表现证实了模型的性能。



## Usage

### Enviroment Requirement
你可以参考environment.yaml中的环境进行配置。

### Implement
下面是TriSSR使用beauty数据集进行训练测试的指令实例：

```shell
python main.py
```
## Citation



## Acknowledgment

This project is based on [Mamba](https://github.com/state-spaces/mamba) and [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their excellent works.

More updates will be posed in the near future! Thank you for your interest.



