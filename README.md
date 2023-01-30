# 基于LSTM的文本分类系统设计

数据来源于中国软件杯，数据量较少。共包含14534条数据，9分类包括包括：财经 、房产 、教育、科技、军事、汽车、体育、游戏和娱乐。使用jieba库进行分词，限制输入特征为600。本项目没有针对模型详细调整参数，只设置了学习率衰减并训练200个epoch，使用Pytorch框架在测试集上可以达到0.86的准确率，而使用Tensorflow框架在测试集上可以达到0.92的准确率，测试集上损失函数均收敛至0.04。

[本项目文章链接](https://yxcai.top/index.php/2023/01/29/%e5%9f%ba%e4%ba%8elstm%e7%9a%84%e6%96%87%e6%9c%ac%e5%88%86%e7%b1%bb%e7%b3%bb%e7%bb%9f%e8%ae%be%e8%ae%a1/)

[GitHub项目链接](https://github.com/hhhhpaaa/Text_classification)
