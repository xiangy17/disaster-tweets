# disaster-tweets

- ## 数据来源：
     kaggle：[https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data](https://www.kaggle.com/competitions/nlp-getting-started/data)

- ## 目标：
     判断测试集中的推文样本是否为灾难推文

- ## 使用：
     直接用program.py

- ## 项目流程精简总结：

1. 数据加载与探索 (EDA)

   读取 train.csv 和 test.csv，查看数据维度、缺失值。对 keyword 和 location 的缺失值填充空字符串。分析 target 分布，以及不同 keyword 下的灾难发生比例（使用 groupby 和 countplot）。
   
3. 特征工程 (Feature Engineering)
 
   基于文本构造统计特征：词数 (word_count)、去重词数 (unique_word_count)、停用词数 (stop_word_count)、URL数 (url_count)、字符数 (char_count)、井号标签数 (hashtag_count)、@提及数 (mention_count)。 使用 seaborn.distplot 可视化这些特征在训练集/测试集、灾难/非灾难类别下的分布差异。

4. 数据清洗 (Data Cleaning)

   找出并剔除标签冲突的数据（同一文本有多个不同 target 的垃圾数据）。

5. 模型预处理 (Preprocessing & Tokenization)
   
   将 keyword 与 text 拼接作为最终文本 final_text。设置随机种子保证可复现性，配置 GPU 设备。使用 BertTokenizer (bert-base-uncased) 将文本转换为 token ids，固定长度 max_length=100，包含 padding 和 special tokens。

6. 模型构建与训练 (Training)
    
   使用 PyTorch 构建 Dataset 和 DataLoader，按 8:2 划分训练集和验证集。加载预训练模型 BertForSequenceClassification (num_labels=2)。使用 AdamW 优化器，设置学习率 (2e-5) 和带 Warmup 的线性学习率调度器 (get_linear_schedule_with_warmup)。训练 4 个 Epoch，包含前向传播、损失计算、反向传播、梯度裁剪（防止梯度爆炸）、参数更新。在验证集上评估 Loss 和 Accuracy。
   
7. 预测与提交 (Inference & Submission)

   使用训练好的模型对测试集进行 no_grad 推理。使用 argmax 获取最终类别预测结果，生成 submission.csv 提交。

---

