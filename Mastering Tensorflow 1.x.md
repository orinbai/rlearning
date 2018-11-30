## Mastering Tensorflow 1.x

### Tensorflow 101

#### TensorFlow 的三个模型

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

#### Tensors

Tensors 是基本计算元素，和TensorFlow中的基础数据结构。一般来讲，在学习TensorFlow我们也仅仅需要学习这一个数据结构。Tensors可以通过以下方式创建：

* 定义常数、操作和变量，并将值传给他们的构造器
* 定义占位符，并将值传至Session.run()
* 通过tf.convert_to_tensor()函数，将诸如标量值、列表和Numpy数组这些Python对象转化

#### Constants

通过tf.constant()常数赋值tensors，语法如下：

``` python
tf.constant(
value,
dtype=None,
shape=None,
name='Const',
verify_shape=False
)
```

``` python
c1 = tf.constant(5, name='x')
c2 = tf.constant(6.0, name='y')
c3 = tf.constant(7.0, tf.float32, name='z')
```

``` python
>>> print('run([c1, c2, c3]) :', tfs.run([c1, c2, c3]))
run([c1, c2, c3]) : [5, 6.0, 7.0]
```

#### Operations

TensorFlow 为我们提供了很多可以用于Tensors的操作。操作符通过传递值并将输出赋值给其他tenser定义。

``` python
op1 = tf.add(c2, c3)
op2 = tf.multiply(c2, c3)
```

```python
>>> print("op1: ", op1);print("op2: ", op2)
op1:  Tensor("Add:0", shape=(), dtype=float32)
op2:  Tensor("Mul:0", shape=(), dtype=float32)
```

```python
>>> print("run(op1): ", tfs.run(op1));print("run(op2): ", tfs.run(op2))
run(op1):  13.0
run(op2):  42.0
```

| 操作符类型       | 操作符                                                       |
| ---------------- | ------------------------------------------------------------ |
| 算数操作符       | tf.add, tf.subtract, tf.multiply, tf.scalar_mul, tf.div, tf.divide, tf.truediv, tf.floordiv, tf.realdiv, tf.truncatediv, tf.floor_div, tf.truncatemod, tf.floormod, tf.mod, tf.cross |
| 基础数学操作符   | tf.add_n, tf.abs, tf.negative, tf.sign, tf.reciprocal, tf.square, tf.round, tf.sqrt, tf.rsqrt, tf.pow, tf.exp, tf.expm1, tf.log, tf.log1p, tf.ceil, tf.floor, tf.maximum, tf.minimum, tf.cos, tf.sin, tf.lbeta, tf.tan, tf.acos, tf.asin, tf.atan, tf.lgamma, tf.digamma, tf.erf, tf.erfc, tf.igamma, tf.squared_diffenrenct, tf.igammac, tf.zeta, tf.polygamma, tf.betainc, tf.rint |
| 矩阵数学操作符   | tf.diag, tf.diag_part, tf.trace, tf.transpose, tf.eye, tf.matrix_diag, tf.matrix_diag_part, tf.matrix_band_part, tf.matrix_set_diag, tf.matrix_transpose, tf.matmul, tf.norm, tf.matrix_determinant, tf.matrix_inverse, tf.cholesky, tf.cholesky_solve, tf.matrix_solve, tf.matrix_triangular_solve, tf.matrix_solve_ls, tf.qr, tf.self_adjoint_eig, tf.self_adjoint_eigvals, tf.svd |
| tensor数学操作符 | tf.tensordot                                                 |
| 复数操作符       | tf.complex, tf.conj, tf.imag, tf.real                        |
| 文本操作符       | tf.string_to_hash_bucket_fast, tf.string_to_hash_bucket_strong, tf.as_string, tf.encode_base64, tf.decode_base64, tf.reduct_join, tf.string_join, tf.string_split, tf.substr, tf.string_to_hash_bucket |

#### 占位符











