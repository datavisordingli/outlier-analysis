# 异常检测
异常检测是用来检测到某些数据不同于其它大多数数据的一种技术。异常检测现在广泛使用在信用卡欺诈，入侵检测，医疗诊断，地球科学等领域。异常检测算法的输出一般是一个评分(score)或者是否异常值的判断(Binary label)。

>>>>>>>>>>>>>>>>>正常数据，噪音和异常数据

一开始是正常的数据，渐渐的，进入噪音区域，如果再往右，进入异常区域。有的时候，我们也把噪音称作弱异常(weak outlier)。
# 模型是关键
本质上来说，在异常检测中，我们首先应该定义一个模型，然后计算出正常数据生成的模型。再用这个模型检测数据，如果数据偏离比较大， 我们就确定它为异常值。
但是在这里选择怎样的数据模型是一个难题。我们是选择高斯模型，线性模型，还是临近模型？每个模型的选择都和实际异常数据相关，也和业务相关(业务方对异常值的定义)。

比如著名的异常检测方法[Z-value test](https://en.wikipedia.org/wiki/Z-test)，但是我们要注意到，使用这个方法的前提是数据是符合正态分布的。

那我们有没有办法自动选择模型，答案是否定的。异常检测很大程度上是一个无监督学习的问题。因为现实中，几乎95%以上的数据都是无标注的，而且我们不能简单的把数据标注为(好-坏)的方式。所以如果你的异常检测模型要通用，那么无监督的异常检测是需要的。好消息是，几乎每个有监督模型都对应一个无监督模型，坏消息是，实际使用无监督模型实在是太难太难了。

| 有监督模型 | 无监督模型 | 类型 |
|---|---|---|
| k-nearest neighbor | k-NN distance, LOF, LOCI | Instance based
| Liner Regression | Principal Component Analysis | Explicit Generalization
| Navie Bayes | Expectation-maximization | Explicit Generalization
| Rocchio | Mahalanobis method, Clustering | Explicit Generalization
| Decision Trees, Random Forest | Isolation Trees, Isolation Forests | Explicit Generalization
| Rules-based | FP-Outlier | Explicit Generalization
| Support vector machine | one class support vector machine | Explicit Generalization
| Neural network | Replicator neural network | Explicit Generalization
| Matrix factorization (incomplete data prediction) | Principal Component Analysis, Matrix factorization | Explicit Generalization


# 异常检测的模型
下面简要介绍一下异常检测里面的模型。详细介绍在以后章节里面会有。影响模型选择的因素有数据类型，数据大小，能否获取异常数据，模型的可解释性等等。
可解释性在模型里面非常的重要，不同的模型可解释性都不一样。一般说来，如果使用的原始特征，并且数据没有转换过(比如PCA)，那么可解释性会强一点，否则可解释性会变弱。

1. 异常检测里的特征值选取 </p>
    因为异常检测模型里面的无监督特性，特征选取是一个非常困难的事情，因为没有一个label说明选择或者不选择某些特征会让模型变好或者变坏。一个可能的做法是 [Kurtosis measure](https://en.wikipedia.org/wiki/Kurtosis): 首先取得数据的平均值u 和方差o, 然后计算 z = (x(i) - u) / z。我们注意到这也是一个分布，并且z的开放和为1。所以这个分布解释了，如果有比较多的异常值，那么Kurtosis measure会比较大。
2. 极值的选取 </p>
    极值的选取本质上是选取统计分布上的长尾，而不是最大最小值。比如考虑下面这个序列[1,2,2,50,98,98,99],这个序列的平均值是50。然而我们仔细观察这个序列，它可以分为两类: [1,2,2] 和 [98,98,99]。所以异常值应该是50。
3. 概率和统计模型 </p>
    优点：几乎适用于任何数据类型，条件是只要符合这个模型。 </p>
    缺点: 首先要确定用哪种分布，这不一定容易做。而且，对于有的模型，因为比较简单，容易产生过拟合。
4. 线性模型 </p>

5. 临近模型(Proximity Based Models) </p>
    临近模型基于这样一个假设，点和点之间有距离，如果相似的点，则距离相近。如果异常点，则距离相远。典型的比如nearest-neighbour模型。
6. 信息理论模型(Information Theoretic Models) </p>
    信息可以生成信息的摘要，如果有异常值，那么这个摘要可能会变得很长。(也可以用信息墒来表示)
    比如，如下两个自字符串:

    ABABABABABABABABABAB
    
    ABACABABABABABABABAB
    </p>
    第一个字符串可以写成 AB*10，也就是10个AB。但是第二个不能这样写，因为有C，所以我们称C为这里的异常值。
7. 高维异常检测 </p>
    高维的异常检测特别难，特别是基于距离的检测(临近模型)，因为在高维，几乎所有的点的距离都是一样的。

# 异常检测的集成方法(Outlier Ensembles)
1. 顺序的集成方法(Sequential Ensembles)
2. 独立的集成方法(Independent Ensembles)

# 数据分析的基本类型
1. 分类，文本和混合数据类型
2. 相互关联的数据类型 </p>
    21. 时间序列和流数据 </p>
    22. 离散序列 </p>
    23. 地理数据 </p>
    24. 网络和图数据 </p>

# 有监督的异常检测

# 异常分析的评估技术
1. 解读 ROC 和 AUC
2. 基准值(Benchmark)的常见错误 </p>
    一种常见的错误是，预设了模型的参数。比如用k-nearest neighbour, 我们假定k=4，然后计算分类。这种做法是不对的，对于一个无监督的数据来说，我们不知道k的具体值。比较合理的做法是，我们给出一个k的范围，比如[3-10]，我们都计算出来，然后选取最好的结果。
