# Mastering Tensorflow 1.x

## Tensorflow 101

### TensorFlow 的三个模型

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

#### 张量

Tensors 是基本计算元素，和TensorFlow中的基础数据结构。一般来讲，在学习TensorFlow我们也仅仅需要学习这一个数据结构。Tensors可以通过以下方式创建：

* 定义常数、操作和变量，并将值传给他们的构造器
* 定义占位符，并将值传至Session.run()
* 通过tf.convert_to_tensor()函数，将诸如标量值、列表和Numpy数组这些Python对象转化

#### 常数

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

#### 操作符

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

占位符是先创建tensor然后在运行时再传值，语法如下：

``` python
tf.placeholder(
dtype,
shape=None,
name=None
)
```

```python
p1 = tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
>>> print("p1: ", p1);print("p2: ", p2)
('p1: ', <tf.Tensor 'Placeholder:0' shape=<unknown> dtype=float32>)
('p2: ', <tf.Tensor 'Placeholder_1:0' shape=<unknown> dtype=float32>)
op4 = p1 * p2
>>> print("run(op4, {p1:2.0, p2:3.0}): ", tfs.run(op4, {p1:2.0, p2:3.0}))
('run(op4, {p1:2.0, p2:3.0}): ', 6.0)
```

也可以使用Python字典——feed_dict——给run()函数传值

```python
>>> print("run(op4, feed_dict={p1:3.0, p2:4.0}): ", tfs.run(op4, feed_dict={p1:3.0, p2:4.0}))
('run(op4, feed_dict={p1:3.0, p2:4.0}): ', 12.0)
```

最后也可以传入向量进行计算

```python
>>> print("run(op4, feed_dict={p1:[2.0, 3.0, 4.0], p2:[3.0, 4.0, 5.0]}): ", tfs.run(op4, feed_dict={p1:[2.0, 3.0, 4.0], p2:[3.0, 4.0, 5.0]}))
('run(op4, feed_dict={p1:[2.0, 3.0, 4.0], p2:[3.0, 4.0, 5.0]}): ', array([  6.,  12.,  20.], dtype=float32))
```

#### 由Python对象来创建Tensor

可以使用tf.convert_to_tensor来由Python对象比如Numpy数组或者列表来创建tensor，其语法如下：

```python
tf.convert_to_tensor(
value,
dtype=None,
name=None,
preferred_dtype=None
)
```

1. 创建一个0维Tensor

   ```python
   tf_t = tf.convert_to_tensor(5.0, dtype=tf.float64)
   >>> print('tf_t: ', tf_t);print('run(tf_t): ', tfs.run(tf_t))
   ('tf_t: ', <tf.Tensor 'Const:0' shape=() dtype=float64>)
   ('run(tf_t): ', 5.0)
   ```

2. 创建一个1-D Tensor

   ```python
   a1dim = np.array([1,2,3,4,5.99])
   >>> print("shape of a1dim: ", a1dim.shape)
   ('shape of a1dim: ', (5,))
   tf_t = tf.convert_to_tensor(a1dim, dtype=tf.float64)
   >>> print("tf_t: ", tf_t)
   ('tf_t: ', <tf.Tensor 'Const_1:0' shape=(5,) dtype=float64>)
   >>> print("tf_t[0]: ", tf_t[0])
   ('tf_t[0]: ', <tf.Tensor 'strided_slice:0' shape=() dtype=float64>)
   >>> print("tf_t[2]: ", tf_t[2])
   ('tf_t[2]: ', <tf.Tensor 'strided_slice_1:0' shape=() dtype=float64>)
   >>> print('run(tf_t): \n', tfs.run(tf_t))
   ('run(tf_t): \n', array([ 1.  ,  2.  ,  3.  ,  4.  ,  5.99]))
   ```

3. 创建一个2-D Tensor

   ```python
   a2dim = np.array([(1,2,3,4,5.99),
                    (2,3,4,5,6.99),
                    (3,4,5,6,7.99)
                    ])
   >>> print('a2dim shape: ', a2dim.shape)
   ('a2dim shape: ', (3, 5))
   tf_t = tf.convert_to_tensor(a2dim, dtype=tf.float64)
   >>> print('tf_t: ', tf_t);print('tf_t[0][0]: ', tf_t[0][0]);print('tf_t[1][2]: ', tf_t[1][2]);print('run(tf_t): ', tfs.run(tf_t))
   ('tf_t: ', <tf.Tensor 'Const_2:0' shape=(3, 5) dtype=float64>)
   ('tf_t[0][0]: ', <tf.Tensor 'strided_slice_3:0' shape=() dtype=float64>)
   ('tf_t[1][2]: ', <tf.Tensor 'strided_slice_5:0' shape=() dtype=float64>)
   ('run(tf_t): ', array([[ 1.  ,  2.  ,  3.  ,  4.  ,  5.99],
          [ 2.  ,  3.  ,  4.  ,  5.  ,  6.99],
          [ 3.  ,  4.  ,  5.  ,  6.  ,  7.99]]))
   ```

4. 创建一个3-D Tensor

   ```python
   a3dim = np.array([[[1,2], [3,4]],
                    [[5,6], [7,8]]
                    ])
   tf_t = tf.convert_to_tensor(a3dim)
   >>> print ('a3dim shape: ', a3dim.shape);print('tf_t: ', tf_t);print('tf_t[0][0][0]: ', tf_t[0][0][0]);print('tf_t[1][1][1]: ', tf_t[1][1][1]);print('run(tf_t): ', tfs.run(tf_t))
   ('a3dim shape: ', (2, 2, 2))
   ('tf_t: ', <tf.Tensor 'Const_3:0' shape=(2, 2, 2) dtype=int64>)
   ('tf_t[0][0][0]: ', <tf.Tensor 'strided_slice_8:0' shape=() dtype=int64>)
   ('tf_t[1][1][1]: ', <tf.Tensor 'strided_slice_11:0' shape=() dtype=int64>)
   ('run(tf_t): ', array([[[1, 2],
           [3, 4]],
   
          [[5, 6],
           [7, 8]]]))
   ```

#### 变量

在TensorFlow中，变量是能保存在执行过程中被修改的值的张量对象。它与占位符很相似，具体区别如下：

| 占位符(tf.placeholder)                         | 变量(tf.Variable)             |
| ---------------------------------------------- | ----------------------------- |
| tf.placeholder定义的是并不随时间改变的输入数据 | tf.Variable的值会随时间变化   |
| tf.placeholder在定义时并不需要初始值           | tf.Variable在定义时需要初始值 |

在TensorFlow中，可以使用tf.Variable来创建变量。以线性模型为例：

$y = W\times{x} + b$

```python
w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
#x是占位符
x = tf.placeholder(tf.float32)
y = w * x + b
>>> print('w: ', w);print('x: ', x);print('b: ',b);print('y: ',y)
('w: ', <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>)
('x: ', <tf.Tensor 'Placeholder_2:0' shape=<unknown> dtype=float32>)
('b: ', <tf.Variable 'Variable_1:0' shape=(1,) dtype=float32_ref>)
('y: ', <tf.Tensor 'add:0' shape=<unknown> dtype=float32>)
```

在TensorFlow的session，我们使用变量前，这些变量必须被初始化。有几种初始化变量的方法：

1. 使用他自己的初始化操作符

   ```python
   tfs.run(w.initializer)
   ```

2. 使用TensorFlow提供的便捷函数来初始化所有变量

   ```python
   tfs.run(tf.global_variables_initializer())
   ```

也可以使用不在run函数中的方式来：

```python
tf.global_variables_initializer().run()
```

还可以使用tf.variables_initializer()来初始化一部分变量。

```python
>>> print('run(y, {x:[1,2,3,4]}): ', tfs.run(y, {x:[1,2,3,4]}))
('run(y, {x:[1,2,3,4]}): ', array([ 0.        ,  0.30000001,  0.60000002,  0.90000004], dtype=float32))
```

#### 从库函数中生成tensor

tensor还可以从TensorFlow的库函数中生成，这些tensor可以或者赋值给变量、常数，或者在初始化时提供给其构造器。比如，我们可以生成100个0的向量:

```python
a = tf.zeros(100,)
>>> print('a: ', tfs.run(a))
('a: ', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32))
```

TensorFlow 提供了不同的函数类型来在张量定义时将其填充：

* 用相同的值填充所有元素
* 用序列填充元素
* 用随机概率分布来填充元素，比如正态分布或者均匀分布











