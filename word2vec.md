## word2vec入门

[秒懂词向量Word2vec的本质](https://zhuanlan.zhihu.com/p/26306795)

NLP里面，最细粒度的是词语，词语组成句子，句子再组成段落、篇章、文档。所以，处理NLP问题，首先就要拿词语开刀。

词语是符号化的形式，但计算机只能接受数值型输入，所以，我们要把他们转换成数值形式，或者说——嵌入到一个数学空间里，这种嵌入方式，就叫词嵌入（word embedding），而word2vec，就是词嵌入的一种。

大部分有监督机器学习模型，都可以总结为：
$$
f(x) -> y
$$
在NLP中，把x看做一个句子里的一个词语，y是这个词语的上下文，那么这里的f，便是NLP里面常出现的**语言模型**，这个模型的目的，就是判断（x，y）这个样本是否符合自然语言的规则，更通俗的说就是，词语x和词语y放在一起是否人话。

word2vec正式源于这个思想，但它的最终目的，不是把f训练的多么完美，而是只关心模型训练后的副产物——模型参数（这里特指神经网络权重），并将这些参数，作为输入x的某种向量化的表示，这个向量叫做——词向量。

### skip-gram和CBOW模型

上面我们提到了语言模型：

- 如果是一个词语作为输入，来预测它周围的上下文，那么这个模型就叫做skip-gram模型
- 如果是一个词语的上下文作为输入，来预测这个词语本身，则是连续词袋（CBOW）模型

#### skip-gram和CBOW的简单情形

最简答一个例子：如上所述，y是x的上下文，所以y只取上下文里一个词语的时候，语言模型就变成：

> 用当前词语预测它的下一个词语

但如上述所说，一般的数学模型只接受数值型输入，这里的x该怎么表示呢？

显然不能用word2vec，因为word2vec是我们训练完模型的产物，现在我们需要一个x的原始输入模式。

答案是：**one-hot encoder**

所谓one-hot encoder，本质上，是用一个只含一个1，其他都是0的向量来唯一表示词语。

接下来，再看skip-gram模型的网络结构，x就是上面提到的one-hot encoder形式的输入，y是在这V个词语上输出的概率，我们希望跟真实的y的one-hot encoder一样。

![v2-a1a73c063b32036429fbd8f1ef59034b_hd](assets/v2-a1a73c063b32036429fbd8f1ef59034b_hd.jpg)

首先说明一点：**隐层的激活函数其实是线性的**，相当于没做任何处理（这也是word2vec简化之前语言模型的独到之处），我们要训练这个神经网络，用**反向传播算法**，本质上是链式求导。

**word2vec精髓**：当模型训练完后，最后得到的其实是神经网络权重，比如现在输入一个x的one-hot encoder，在输入层到隐藏层的权重里，只有对应1这个位置的权重被激活，这些权重的个数，跟隐含层节点数是一致的，从而这些权重组成一个响亮vx来表示x，而因为每个词语的one-hot encoder里面1的位置是不同的，所以这个向量vx就可以用来唯一表示x。

#### **skip-gram更一般的情形**

上述讨论的是最简单的skip-gram情形，即y只有一个词语。

当y为多个词语时，网络结构如下：

![v2-ca81e19caa378cee6d4ba6d867f4fc7c_hd](assets/v2-ca81e19caa378cee6d4ba6d867f4fc7c_hd.jpg)

> 可以看成是 单个 x -> 单个y模型的并联，cost function是单个cost function的累加（取log之后）。

#### CBOW更一般的情形

跟skip-gram相似，只不过：

> skip-gram是预测一个词语的上下文，而CBOW是用上下文预测这个词

网络结构如下：

![v2-d1ca2547dfb91bf6a26c60782a26aa02_hd](assets/v2-d1ca2547dfb91bf6a26c60782a26aa02_hd.jpg)

跟skip-gram模型的并联不同，这里是输入变成了多个单词，所以要**对输入处理**下（一般是求和然后平均），输出的cost function不变。

### word2vec的训练trick

**hierarchical softmax**与**negative sampling**，是word2vec的训练技巧。hierarchical softmax是softmax的一种近似形式，negative sampling也是从其他方法借鉴而来的。

为什么要用训练技巧呢？如我们刚才所提，word2vec本质上是一个语言模型，它的输出节点数是V个，对应了V个词语，本质上是一个多分类问题。但实际中，由于V的个数非常大，会给计算带来很大困难。所以，需要训练技巧来加速训练。

- hierarchical softmax，本质是把N分类问题变成log(N)次二分类
- negative sampling，本质是预测总体类别的一个子集

### 实战

真正应用时，可以通过一个Python第三方库的接口，**Gensim**。但对理论探究仍有必要，可以更好地知道参数的意义、模型结果受哪些因素影响，以及举一反三应用到其他问题中，甚至更改源码以实现定制化需求。

[使用Gensim实现Word2Vec和FastText词嵌入](https://zhuanlan.zhihu.com/p/59860985)

```python
from gensim.models import Word2Vec
model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)
```

- sentences：切分句子的列表。
- size：嵌入向量的维数
- window：你正在查看的上下文单词数
- min_count：告诉模型忽略总计数小于这个数字的单词。
- workers：正在使用的线程数
- sg：是否使用skip-gram或CBOW

