## Mastering Tensorflow 1.x

### Tensorflow 101

Tensorflow有三个模型：

* 数据模型：数据模型由tensor组成。就是Tensorflow程序中数据单位的增加，操作和存储。
* 程序模型：程序模型由数据流图或者计算图组成。Tensorflow中创造一个程序就是建立一个或者多个Tensorflow的计算图。
* 执行模型：执行模型是根据输入值，从初始节点开始，依据依赖条件各个节点依次触发。这种触发仅依赖于输入。

在项目中使用Tensorflow，也就是学习使用Tensorflow的API来编码。Tensorflow有很多API，基本分为两类：

1. 底层API（lower-level API），也被称为Tensorflow Core，提供了精确的底层函数，因此提供了模型中使用和实现lib的完全控制。
2. 高层API，这部分lib提供了高级API，可以相对简单的学习和实现模型。比如，TF Estimators，TFLearn，TFSlim，Sonnet和Keras。

tensorflow有两种Session，Session和InteractiveSession

``` python
import tensorflow as tf

tfs = tf.InteractiveSession()
hello = tf.constant("Hello, TensorFlow!!")
print(tfs.run(hello))
```

