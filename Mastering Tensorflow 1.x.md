---
typora-copy-images-to: media
---

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

TensorFlow 为我们提供了很多可以用于Tensors的操作。操作符通过传递值并将输出赋值给其他tensor定义。

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

$$
y=W\times x + b
$$

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

##### 用相同的值填充tensor元素

| tensor生成函数                                          | 描述                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| zeros(shape, dtype=tf.float32, name=None)               | 根据给定的shape创建tensor，所有元素均为0                     |
| zeros_like(tensor, dtype=None,name=None, optimize=True) | 创建一个参数中tensor同样shape的tensor，所有元素均为0         |
| ones(shape, dtype=tf.float32, name=None)                | 创建给定shape的tensor，所有元素均为1                         |
| ones_like(tensor, dtype=None, name=None, optimize=True) | 同zeros_like                                                 |
| fill(dims, value, name=None)                            | 根据参数dims创建一个同样shape的tensor，所有元素设定为参数value的值 |

##### 用序列填充tensor元素

| Tensor生成函数                                           | 描述                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| lin_space(start, stop, num, name=None)                   | 创建一个tensor，其值是为[start, stop]范围内的num数量的数值。数据类型为start的参数。 |
| range([start], limit, delta=1, dtype=None, name='range') | 生成一个1-D tensor，其值是在[start, limit]中的数据序列，按照delta的值增加。如果dtype没有指定，则与start相同。如果start被忽略，那么从0开始。 |

##### 使用随机分布填充tensor元素

| Tensor生成函数                                               | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) | 生成指定shape的tensor，用正态分布填充其值:normal（mean, stddev)。 |
| truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None) | 生成指定shape的tensor，使用==截断正态分布==填充其值:normal(mean, stddev)。==截断==是指其值总是小于距均值2个标准差的数字。 |
| random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None) | 生成指定shape的tensor，使用均匀分布填充其值：uniform(minval, maxval) |
| random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None) | 生成制定shape的tensor，使用伽玛分布填充其值：gamma(alpha, beta)。 |

更多细节可以查询：[seed 相关](https://www.tensorflow.org/api_docs/python/tf/set_random_seed) [gamma函数](https://www.tensorflow.org/api_docs/python/tf/random_gamma)

##### 通过tf.get_variable() 获取变量

如果我们对一个已经定义过的变量再定义，那么TensorFlow会抛出异常。因此，我们应该通过tf.get_variable()来替代tf.Variable()。这个函数在变量存在时会返回一个同名的已存在变量，否则创建这个变量。

在分布式TensorFlow中，tf.get_variable()会获得全局变量，如果要获得局部变量要使用tf.get_local_variable()

```python
w = tf.get_variable(name='w', shape=[1], dtype=tf.float32, initializer=[.3])
b = tf.get_variable(name='b', shape=[1], dtype=tf.float32, initializer=[-.3])
```

*书中有误，tf.get_variable() 去获得已存在变量时会报错，需要在变量域内设置reuse=True*

所以，如果要重用已定义变量，那么需要设置tf.variable_scope.reuse_variable()或者设置tf.variable.scope(resue=True)。

### 数据流图或者计算图

数据流图或者计算图是TensorFlow中的基本计算单位，之后我们会称之为计算图。其中，每个节点是一个TensorFlow的操作符(tf.Operation)，每个边是一个在节点间变换的tensor(tf.Tensor)。

TensorFlow中的一段程序基本上就是一个计算图，创建一个计算图，其节点表示变量、常熟、占位符和操作符，然后将其灌入TensorFlow。TensorFlow寻找可以激发或者执行的第一批节点，然后这些节点的激发会带来其他节点的激发，以此类推。

TensorFlow 由一个默认图开始，除非有显示的指定图，否则新节点将被隐式的加入默认图中。我们可以通过一下命令获得默认图：

```python
graph = tf.get_default_graph()
```

如果我们要定义三个输入值并将其相加后输出
$$
y = x_1 + x_2 + x_3
$$
那么其计算图会类似下面的图：

``` mermaid
graph LR
var1((x1))-->op((add))
var2((x2))-->op((add))
var3((x3))-->op((add))
op((add))-->var4((y))


```

在TensorFlow中，看起来是这样的：

```python
y = tf.add(x1+x2+x3)
```

我们创建并执行以下计算图：
$$
y = W\times x + b
$$

```python
w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
y = w*x + b
output = 0
with tf.Session() as tfs:
    tf.global_variables_initializer().run()
    output = tfs.run(y, {x: [1, 2, 3, 4]})
2018-12-04 02:30:05.792269: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1306] Adding visible gpu devices: 0
2018-12-04 02:30:05.981830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:987] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3103 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
>>> print("output: ", output)
('output: ', array([ 0.        ,  0.30000001,  0.60000002,  0.90000004], dtype=float32))
```

#### 执行顺序与惰性加载

节点按照依赖顺序执行。如果节点a依赖节点b，那么当请求执行b时，a将先于b执行。节点对象当其需要的时候才会被创建和初始化，被成为惰性加载。

当我们需要控制执行顺序时，可以使用tf.Graph.control_dependencies()改变顺序。

```python
with graph_variable.control_dependencies([c, d]):
    # other statements here
```

假设我们有a、b、c、d四个节点，上面的方法将保证在代码块内所有的节点执行都会在c、d执行之后。

#### 通过计算设备（CPU、GPU）运行图

一个图可以被分成多个部分，每个部分可以被指派并在不同的独立设备上进行执行，比如CPU或者GPU。下面命令可以列出能够运行图的设备：

```python
from tensorflow.python.client import device_lib
>>> print(device_lib.list_local_devices())
2018-12-04 03:01:18.691585: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1306] Adding visible gpu devices: 0
2018-12-04 03:01:18.691977: I tensorflow/core/common_runtime/gpu/gpu_device.cc:987] Creating TensorFlow device (/device:GPU:0 with 3106 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 10858870713487053697
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 3257794560
locality {
  bus_id: 1
}
incarnation: 4472897408438210799
physical_device_desc: "device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1"
]
```

在TensorFlow中，通过字串 /device:<device_type>:<device_idx>来识别设备。信息中的CPU、GPU就是device_type，0是device_idx(device index)。

无论是几颗CPU都显示一个，因为TensorFlow隐式的会在CPU上进行分布，所以CPU0表示了TensorFlow可以使用的所有CPU资源。当TensorFlow开始运行图时，它将每个图的独立部分运行在不同的线程上，而每个线程也运行在不同的CPU上。我们可以通过改变inter_op_parallelism_threads的值来限制线程的数量。类似的，在独立部分中，操作也有多线程的能力。TensorFlow会把特定的操作多线程化，这部分线程池的数量可以通过intra_op_parallelism_threads的值来改变。

#### 将图节点分配在特定的计算设备上

我们可以通过一个设置对象来打开变量分配记录，设置 log_device_placement 的属性为 true，并将这个config对象传给后续的操作：

```python
tf.reset_default_graph()

w = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
y = w * x + b

config = tf.ConfigProto()
config.log_device_placement=True

with tf.Session(config=config) as tfs:
    tf.global_variables_initializer().run() #原书代码错误global_variables_initializer是tf的函数
    print('output: ', tfs.run(y, {x: [1,2,3,4]}))
... 
2018-12-04 03:23:02.994866: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1306] Adding visible gpu devices: 0
2018-12-04 03:23:02.995237: I tensorflow/core/common_runtime/gpu/gpu_device.cc:987] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3101 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1
2018-12-04 03:23:02.995488: I tensorflow/core/common_runtime/direct_session.cc:298] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1

Variable_1: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.000772: I tensorflow/core/common_runtime/placer.cc:875] Variable_1: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
Variable_1/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.000835: I tensorflow/core/common_runtime/placer.cc:875] Variable_1/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
Variable_1/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.000866: I tensorflow/core/common_runtime/placer.cc:875] Variable_1/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
Variable: (VariableV2): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.000898: I tensorflow/core/common_runtime/placer.cc:875] Variable: (VariableV2)/job:localhost/replica:0/task:0/device:GPU:0
Variable/read: (Identity): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.000948: I tensorflow/core/common_runtime/placer.cc:875] Variable/read: (Identity)/job:localhost/replica:0/task:0/device:GPU:0
mul: (Mul): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.000992: I tensorflow/core/common_runtime/placer.cc:875] mul: (Mul)/job:localhost/replica:0/task:0/device:GPU:0
add: (Add): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.001024: I tensorflow/core/common_runtime/placer.cc:875] add: (Add)/job:localhost/replica:0/task:0/device:GPU:0
Variable/Assign: (Assign): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.001055: I tensorflow/core/common_runtime/placer.cc:875] Variable/Assign: (Assign)/job:localhost/replica:0/task:0/device:GPU:0
init: (NoOp): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.001082: I tensorflow/core/common_runtime/placer.cc:875] init: (NoOp)/job:localhost/replica:0/task:0/device:GPU:0
Placeholder: (Placeholder): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.001113: I tensorflow/core/common_runtime/placer.cc:875] Placeholder: (Placeholder)/job:localhost/replica:0/task:0/device:GPU:0
Variable_1/initial_value: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.001141: I tensorflow/core/common_runtime/placer.cc:875] Variable_1/initial_value: (Const)/job:localhost/replica:0/task:0/device:GPU:0
Variable/initial_value: (Const): /job:localhost/replica:0/task:0/device:GPU:0
2018-12-04 03:23:03.001169: I tensorflow/core/common_runtime/placer.cc:875] Variable/initial_value: (Const)/job:localhost/replica:0/task:0/device:GPU:0
('output: ', array([ 0.        ,  0.30000001,  0.60000002,  0.90000004], dtype=float32))
```

也可以用tf.device来指定设备

```python
tf.reset_default_graph()
with tf.device('/device:CPU:0'):
    w = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    y = w * x + b

config = tf.ConfigProto()
config.log_device_placement = True
with tf.Session(config=config) as tfs:
    tfs.run(tf.global_variables_initializer())
    print("output: ", tfs.run(y, {x:[1,2,3,4]}))
    
... 
2018-12-04 03:29:55.710296: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1306] Adding visible gpu devices: 0
2018-12-04 03:29:55.710689: I tensorflow/core/common_runtime/gpu/gpu_device.cc:987] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3103 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1
2018-12-04 03:29:55.711002: I tensorflow/core/common_runtime/direct_session.cc:298] Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1

Variable_1: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728381: I tensorflow/core/common_runtime/placer.cc:875] Variable_1: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
Variable_1/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728453: I tensorflow/core/common_runtime/placer.cc:875] Variable_1/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
Variable_1/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728483: I tensorflow/core/common_runtime/placer.cc:875] Variable_1/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
Variable: (VariableV2): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728513: I tensorflow/core/common_runtime/placer.cc:875] Variable: (VariableV2)/job:localhost/replica:0/task:0/device:CPU:0
Variable/read: (Identity): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728540: I tensorflow/core/common_runtime/placer.cc:875] Variable/read: (Identity)/job:localhost/replica:0/task:0/device:CPU:0
mul: (Mul): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728565: I tensorflow/core/common_runtime/placer.cc:875] mul: (Mul)/job:localhost/replica:0/task:0/device:CPU:0
add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728633: I tensorflow/core/common_runtime/placer.cc:875] add: (Add)/job:localhost/replica:0/task:0/device:CPU:0
Variable/Assign: (Assign): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728665: I tensorflow/core/common_runtime/placer.cc:875] Variable/Assign: (Assign)/job:localhost/replica:0/task:0/device:CPU:0
init: (NoOp): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728691: I tensorflow/core/common_runtime/placer.cc:875] init: (NoOp)/job:localhost/replica:0/task:0/device:CPU:0
Placeholder: (Placeholder): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728719: I tensorflow/core/common_runtime/placer.cc:875] Placeholder: (Placeholder)/job:localhost/replica:0/task:0/device:CPU:0
Variable_1/initial_value: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728745: I tensorflow/core/common_runtime/placer.cc:875] Variable_1/initial_value: (Const)/job:localhost/replica:0/task:0/device:CPU:0
Variable/initial_value: (Const): /job:localhost/replica:0/task:0/device:CPU:0
2018-12-04 03:29:55.728771: I tensorflow/core/common_runtime/placer.cc:875] Variable/initial_value: (Const)/job:localhost/replica:0/task:0/device:CPU:0
('output: ', array([ 0.        ,  0.30000001,  0.60000002,  0.90000004], dtype=float32))
```

#### 简单分配规则

``` python
如果该图之前运行过，那么剩余的节点会分配在之前的设备上;
之外，如果使用了tf.device()，那么节点会分配在指定的设备上;
或者如果GPU可用，那么被分配到第一个GPU上;
如果GPU不可用，那么被分配到CPU上。
```

#### 动态分配

tf.device() 也可以传入一个函数名来代替设备字串，在这种情况下，这函数必须返回一个设备字串。这个特性允许复杂算法将变量分配到不同的设备上。

#### 软分配

当我们将TensorFlow的操作分配到GPU上，TF必须有这种操作的GPU实现，这被成为Kernel。如果Kernel不可用就会报运行错误。或者当该GPU不存在，也会有运行错误。最好的办法，就是允许在GPU不可用时，可以把这种操作分配到CPU上。我们可以使用如下命令：

```python
config.allow_soft_placement = True
```

#### GPU 显存处理

当你开始运行TensorFlow会话时，默认情况下，它会抓走所有的GPU内存，即使你只将操作和变量分配到多GPU系统的一个额GPU上。这时，如果你运行另外一个会话（session）时，就会有一个内存不足的错误。解决办法有几个：

* 对于多GPU系统，设置环境变量 CUDA_VISIBLE_DEVICES = <list of device idx>

  ```python
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  ```

  这个设置之后的代码，将会只用光可见GPU的内存。

* 如果你不希望会话用光所有的显存，那么使用配置选项 per_process_gpu_memory_fraction 来分配内存的百分比

  ```python
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  ```

  上面代码分配50%的所有显存

* 也可以限制TensorFlow进程在处理开始时指获取最小的所需内存，所着处理的深入而不断增长

  ```python
  config.gpu_options.allow_growth = True
  ```

  这个方式仅允许其内存分配增长，但内存用不释放

### 多图

可以从默认图中分离出你自己的图，并在一个会话中执行他们。然而，创建和执行多图并不推荐，因为以下缺点：

* 同一个程序中创建和使用多图需要多个TensorFlow会话，并且每个会话都要消耗大量的资源。
* 在图中不能直接传递数据

推荐方法是在一个图中有多个子图。如果你需要用自己的图替代默认图，可以使用tf.Graph()。

```python
g = tf.Graph()
output = 0
with g.as_default():
    w = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    y = w * x + b

with tf.Session(graph=g) as tfs:
    tf.global_variables_initializer().run()
    output = tfs.run(y, {x: [1,2,3,4]})
 
>>> print('output: ', output)
output:  [0.         0.3        0.6        0.90000004]
```

### TensorBoard

计算图的复杂度非常高，即使是普通规模的问题。大型计算图比如复杂的机器学习模型会变得相当令人困惑并且令人难以理解。可视化会帮助我们理解和解释计算图，同时可以加速debug和优化TensorFlow程序。TensorFlow有一个内建的可视化计算图工具叫做TensorBoard。

#### 最小化TensorBoard的例子

1. 开始定义我们线性模型的变量和占位符

   ```python
   w = tf.Variable([.3], name='w', dtype=tf.float32)
   b = tf.Variable([-.3], name='b', dtype=tf.float32)
   x = tf.placeholder(name='x', dtype=tf.float32)
   y = w * x + b
   ```

2. 初始化一个会话，在其上下文内，完成以下的事：

   * 初始化全局变量
   * 创建 tf.summary.FileWriter 来在tflogs目录中新建默认图事件的输出
   * 有效执行现行模型，获得 y 节点的值

   ```python
   with tf.Session() as tfs:
       tfs.run(tf.global_variables_initializer())
       writer = tf.summary.FileWriter('tflogs', tfs.graph)
       print('run(y, {x:3}) :', tfs.run(y, {x:3}))
   ```

3. 我们获得输出：

   ```python
   run(y, {x:3}) : [0.6]
   ```

随着程序执行，tflog目录里已经记录所有的信息，以备TensorBoard可视化。打开命令行：

```python
tensorboard --logdir='tflogs'
TensorBoard 1.6.0 at http://Master:6006 (Press CTRL+C to quit)
```

#### TensorBoard 细节

TensorBoard 通过读取TensorFlow生成的日志工作。因此，我们需要修改定义好的程序模型来加入额外可以产生我们需要用TensorBoard可视化数据的额外操作节点。TensorBoard需要的程序模型或者程序流，一般可以定义如下：

1. 如常的创建计算图。
2. 创建汇总节点（summary nodes）。将来自tf.summary包中的附加汇总操作符附加到产生我们需要收集和分析数据的节点上。
3. 运行模型节点的同时运行汇总节点。通常使用便捷函数 tf.summary.merge_all() 来将所有的汇总节点合并到一个节点上。运行这个节点基本上就会执行所有的汇总节点。合并的汇总节点会产生序列化的 Summary ProtocolBuffers 对象，包含所有汇总节点的总和。
4. 通过把 Summary ProtocolBuffers 对象传递给函数tf.summary.FileWriter()来把事件日志写入磁盘。
5. 启动TensorBoard并分析可视化数据。

## Keras 101

### Keras中的神经网络模型

Keras中的神经网络模型被定义为层图。Keras中可以使用Sequential或者Functional API来创建模型。因此作为经验总结，使用顺序API创建简单模型，函数API创建复杂模型。不过，通过函数API创建复杂模型有利于将来将模型扩展成复杂模型，甚至分支和分享。所以我们主要用函数API来创建模型。

#### Keras 中创建模型的工作流

Keras中简单的工作流如下：

1. 创建模型
2. 在模型中创建和增加层
3. 编译模型
4. 训练模型
5. 使用模型预测和评估

### 创建Keras模型

#### 用顺序API创建Keras模型

```python
model = Sequential()
```

现在可以给这个模型增加层，也可以在创建的时候就把层的参数传递进去

```python
model = Sequential([Dense(10, input_shape=(256, )),
    Activation('tanh'),
    Dense(10),
    Activation('softmax')
    ])
```

#### 用函数API创建Keras模型

使用函数API，可以创建一个Model类的实例，拥有input和output参数。输入输出参数表现为一个或多个输入输出张量。

```python
model = Model(inputs=tensor1, output=tensor2)
```

上述代码中，tensor1和tensor2可以是张量或者可以作为张量对待的对象。比如，Keras的层。

```python
model = Model(inputs=[i1, i2, i3], output=[o1, o2, o3])
```

也可以传入张量列表。

### Keras 的层

为方便构造网络结构，Keras内建了很多层类别。

#### Keras 核心层

| 层名称                 | 描述                                                         |
| ---------------------- | ------------------------------------------------------------ |
| Dense                  | 就是一个简单的全链接神经网络层。这个曾会产生下面这个函数的输出 activation((inputs × weights) + bias)，activation是指传递给层的激活函数，默认值是None。 |
| Activation             | 本层是针对输出的特定激活函数。其产生下面函数的输出 activation(inputs) activation是传递给层的激活函数。目前激活层有如下的的激活函数：softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid 和 linear |
| Dropout                | 这个层是对输入数据有在特定的丢弃率下正规化丢弃               |
| Flatten                | 这一层扁平化输入，比如，一个三位输入会被扁平化，并产生一维输出 |
| Reshape                | 将输入转换为特定的shape                                      |
| Permute                | 按照特定的模式重新排序输入维度                               |
| RepeatVector           | 根据给定的次数重复输入。比如，如果输入是一个2D张量，其shape为(#samples, #features)，同时本层给定n次重复，那么输出会是3D 张量，其shape为(#samples, n, #features) |
| Lambda                 | 将给出的函数封装为一层。因此，输入被传给给定自定义的函数并产生输出。这层给Keras用户提供了极其方便的方式来将其函数变成层 |
| ActivityRegularization | 对其输入采用L1、L2或者其组合来正规化。这个层被用于激活层或有激活函数的层的输出 |
| Masking                | 当所有输入张量的值等于层中的遮罩值时，遮住或者跳过输入张量的这步 |

#### Keras 卷积层

| 层名            | 描述                                                         |
| --------------- | ------------------------------------------------------------ |
| Conv1D          | 在输入为单一空间或者临时维度上应用卷积                       |
| Conv2D          | 在输入上应用2维卷积                                          |
| SeparableConv2D | 在每个输入通道中应用深度级空间卷积，后随混合起来的点级卷积，一起产生输出通道 |
| Conv2DTranspose | 根据输入shape反转卷积的shape来产生这些卷积                   |
| Conv3D          | 在输入上应用3维卷积                                          |
| Cropping1D      | 根据临时维度裁剪输入数据                                     |
| Cropping2D      | 根据空间维度裁剪输入数据，比如，图片的宽和高                 |
| Cropping3D      | 根据时空裁剪输入数据，所有的三维空间                         |
| UpSampling1D    | 在时间轴上重复输入数据指定次数                               |
| UpSampling2D    | 在行和列两个维度上重复输入数据的两个维指定次数               |
| UpSampling3D    | 沿着三维重复输入数据的三维指定次数                           |
| ZeroPadding1D   | 在时间维开始和结束的地方增加0                                |
| ZeroPadding2D   | 在2D张量的上下左右增加0的行和列                              |
| ZeroPadding3D   | 在3D张量的三个维度上增加0                                    |

Keras池层

| 层名                   | 描述                                 |
| ---------------------- | ------------------------------------ |
| MaxPooling1D           | 对于一维输入数据实现最大池化操作     |
| MaxPooling2D           | 对于二维输入数据实现最大池化操作     |
| MaxPooling3D           | 对于三维输入数据实现最大池化操作     |
| AveragePooling1D       | 对于一维输入数据实现平均池化操作     |
| AveragePooling2D       | 对于二维输入数据实现平均池化操作     |
| AveragePooling3D       | 对于三维输入数据实现平均池化操作     |
| GlobalMaxPooling1D     | 对于一维输入数据实现全局最大池化操作 |
| GlobalMaxPooling2D     | 对于二维输入数据实现全局最大池化操作 |
| GlobalAveragePooling1D | 对于一维输入数据实现全局平均池化操作 |
| GlobalAveragePooling2D | 对于一维输入数据实现全局平均池化操作 |

#### Keras 本地链接层

| 层名               | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| LocallyConnected1D | 在输入数据的单一空间或临时维度上通过对输入的不同批次中应用不同的集合或者过滤器应用卷积，所以并不共享权重 |
| LocallyConnected2D | 对于输入的二维数据通过对输入的不同批次中应用不同的集合或者过滤器应用卷积，所以并不共享权重 |

#### Keras 递归层

| 层名      | 描述                     |
| --------- | ------------------------ |
| SimpleRnn | 实现全连接的递归神经网络 |
| GRU       | 实现门控递归单元网络     |
| LSTM      | 实现长短期记忆网络       |

#### Keras 嵌入网络

| 层名 | 描述                                                         |
| ---- | ------------------------------------------------------------ |
| 嵌入 | 获取由索引组成的shape(batch_size, sequence_length)2D张量，输出由shape(batch_size, sequence_length, output_dim)稠密向量组成的张量 |

#### Keras 融合层

| 层名                                                  | 描述                                       |
| ----------------------------------------------------- | ------------------------------------------ |
| Add                                                   | 计算输入张量的元素级加法                   |
| Multiply                                              | 计算输入张量的元素级乘法                   |
| Average                                               | 计算输入向量的元素级平均                   |
| Maximum                                               | 计算输入向量的元素级最大                   |
| Concatenate                                           | 在指定坐标轴上连接输入张量                 |
| Dot                                                   | 计算两个输入张量的点积                     |
| add, multiply, average, maximum, concatenate, and dot | 这些函数代表这个表格中各自融合层的函数界面 |

#### Keras 高级激活层

| 层名            | 描述                       |
| --------------- | -------------------------- |
| LeakyRelu       | 计算泄漏版本的ReLU激活函数 |
| PReLU           | 计算参数化ReLU激活函数     |
| ELU             | 计算指数线性单元激活函数   |
| ThresholdedReLU | 计算阈版本的ReLU激活函数   |

#### Keras 正规化层

| 层名               | 描述                                                         |
| ------------------ | ------------------------------------------------------------ |
| BatchNormalization | 这层正规化了每个批次中前一层的输出，本层的输出接近0均值1标准差 |

#### Keras 噪声层

这些层可以被加入模型通过增加噪声以防止过拟合;也被称为正则化层。这些层的操作与核心层中的Dropout()和ActivityRegularizer()一致

| 层名            | 描述                                                         |
| --------------- | ------------------------------------------------------------ |
| GaussianNoise   | 将0中心的高斯分布噪音附加到输入                              |
| GaussianDropout | 将1为中心的多种高斯噪音附加到输入                            |
| AlphaDropout    | 丢弃一定百分比的输入，同时丢弃后输出的均值和方差非常匹配输出的均值和方差 |

### 给Keras模型增加层

#### 使用顺序API为Keras模型增加层

使用顺序API，你可以通过实例化前面提及的层类型来创建新层。然后，这些创建的层可以通过model.add()加入模型。比如：

```python
model = Sequential()
model.add(Dense(10, input_shape=(256,)))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('softmax'))
```

#### 使用函数API为Keras模型添加层

使用函数API，层在最开始的时候以函数的方式创建，然后在构建模型时，输入输出层被提供给tensor 参数。比如：

1. 创建输入层：

   ```python
   input = Input(shape=(64,))
   ```

2. 从输入层中以函数的形式创建稠密层

   ```python
   hidden = Dense(10)(input)
   ```

3. 同样的，在前一层之后，以函数的形式创建更多的隐藏层：

   ```python
   hidden = Activation('tanh')(hidden)
   hidden = Dense(10)(hidden)
   output = Activation('softmax')(hidden)
   ```

4. 最后带着输入输出层实例化模型对象：

   ```python
   model = Model(inputs=input, outputs=output)
   ```

### 编译Kares模型

之前建立的模型在用于预测和训练前，需要使用model.compile()进行编译。compile()的使用说明如下：

```python
compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)
```

编译方法需要三个参数：

* optimizer：可以使用自己或者由Keras提供的函数。这个函数是用于更新优化迭代参数。Keras内置的优化函数：
  * SGD
  * RMSprop
  * Adagrad
  * Adadelta
  * Adam
  * Adamax
  * Nadam
* loss：可以使用自己或者提供的损失函数。优化器通过优化参数使得loss函数的输出最小化。Keras提供的损失函数：
  * mean_squared_error
  * mean_absolute_error
  * mean_absolute_pecentage_error
  * mean_squared_logarithmic_error
  * squared_hinge
  * hinge
  * categorical_hinge
  * sparse_categorical_crossentropy
  * binary_crossentropy
  * poisson
  * cosine proximity
  * binary_accuracy
  * categorical_accuracy
  * top_k_categorical_accuracy
  * sparse_top_k_categorical_accuracy
* metrics：列出需要训练时需要收集的指标。如果打开了详细输出模式，每代时会被输出。这些指标与损失函数相似，所有损失函数也可以作为指标函数使用。

### 训练Keras模型

通过运行model.fit()就可以训练Keras模型，说明如下：

```python
fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, suffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
```

详情可以查看[Keras网站](https://keras.io/models/sequential/)

### 使用Keras模型预测

训练好的模型可以使用model.predict()来预测或使用model.evaluate()来进行评估，具体使用说明如下：

```python
predict(self, x, batch_size=32, verbose=0)
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```

### Keras的附加模块

Keras提供了一些附加模块，通过增加的功能性来补足Keras的基本工作流。下面是一些模块：

* preprocessing 模块提供了一些序列、图片和文本数据的预处理函数
* datasets 模块提供了快速操作某些流行数据集的函数，比如 CIFAR10 图片，CIFAR100图片，IMDB 影评， 路透新闻标题，MNIST手写数字和波士顿房屋价格。
* initializers 提供了初始化每层的参数权重的函数，例如Zeros, Ones, Constant, RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Orthogonal, Identity, lecun_normal和he_uniform。
* models 模块提供了一些函数来载入模型架构和权重，比如 model_from_json, model_from_yaml和load_model。模型架构可以是用model.to_yaml()和model.to_json()方法来存储。模型权重可以使用model.save()方法，以HDF5文件存储。
* applications 提供了一些预制或者预训练的模型，例如Xception， VGG16, VGG19, ResNet50, Inception V3， InceptionResNet V2和MobileNet。

### 使用Keras顺序模型为例为 MNIST 数据集建模

```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras import utils
import numpy as np

# 定义超参数
batch_size = 100
n_inputs = 784
n_classes = 10
n_epochs = 10

# 获得数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 调整shape到2维 28×28 像素
# 将图片放进784像素的单个向量
x_train = x_train.reshape(60000, n_inputs)
x_test = x_test.reshape(10000, n_inputs)

# 转换输入值为float32
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# 标准化图片向量值
x_train /=255
x_test /= 255

# 转换输出数据为one hot编码格式
y_train = utils.to_categorical(y_train, n_classes)
y_test = utils.to_categorical(y_test, n_classes)

# 构建顺序模型
model = Sequential()
# 第一层必须指定输入向量的维度
model.add(Dense(units=128, activation='sigmoid', input_shape=(n_input,)))
# 增加Dropout层防止过拟合
model.add(Dropout(0.1))
model.add(Dense(units=128, activation='sigmoid'))
model.add(Dropout(0.1))
# 输出层只能让神经元与输出的数量相等
model.add(Dense(units=n_classes, activation='softmax'))

# 输出我们模型的汇总信息
model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy',
             optimizer=SGD(),
             metrics=['accuracy'])
# 训练模型
model.fit(x_train, y_train,
         batch_size=batch_size,
         epochs=n_epochs)
# 评估模型并打印准确率得分
scores = model.evaluate(x_test, y_test)

print('\n loss:', scores[0])
print('\n accuracy:', scores[1])

Layer (type)                 Output Shape              Param #   
=================================================================
dense_13 (Dense)             (None, 128)               100480    
_________________________________________________________________
dropout_9 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_14 (Dense)             (None, 128)               16512     
_________________________________________________________________
dropout_10 (Dropout)         (None, 128)               0         
_________________________________________________________________
dense_15 (Dense)             (None, 10)                1290      
=================================================================
Total params: 118,282
Trainable params: 118,282
Non-trainable params: 0
_________________________________________________________________

Epoch 1/10
2018-12-06 18:24:51.613219: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-12-06 18:24:51.614108: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1411] Found device 0 with properties: 
name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.531
pciBusID: 0000:03:00.0
totalMemory: 1.95GiB freeMemory: 1.84GiB
2018-12-06 18:24:51.614143: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1490] Adding visible gpu devices: 0
2018-12-06 18:24:52.707358: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-12-06 18:24:52.707414: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977]      0 
2018-12-06 18:24:52.707425: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990] 0:   N 
2018-12-06 18:24:52.707610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1103] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1595 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:03:00.0, compute capability: 6.1)
60000/60000 [==============================] - 4s 62us/step - loss: 2.3023 - acc: 0.1269
Epoch 2/10
60000/60000 [==============================] - 2s 27us/step - loss: 2.2287 - acc: 0.1993
Epoch 3/10
60000/60000 [==============================] - 2s 27us/step - loss: 2.1361 - acc: 0.2947
Epoch 4/10
60000/60000 [==============================] - 2s 27us/step - loss: 1.9934 - acc: 0.3967
Epoch 5/10
60000/60000 [==============================] - 2s 27us/step - loss: 1.7898 - acc: 0.4878
Epoch 6/10
60000/60000 [==============================] - 2s 27us/step - loss: 1.5636 - acc: 0.5584
Epoch 7/10
60000/60000 [==============================] - 2s 27us/step - loss: 1.3600 - acc: 0.6125
Epoch 8/10
60000/60000 [==============================] - 2s 27us/step - loss: 1.1919 - acc: 0.6576
Epoch 9/10
60000/60000 [==============================] - 2s 27us/step - loss: 1.0609 - acc: 0.6941
Epoch 10/10
60000/60000 [==============================] - 2s 27us/step - loss: 0.9583 - acc: 0.7225

10000/10000 [==============================] - 0s 31us/step
loss: 0.8364587327003479
accuracy: 0.7888
```

## TensorFlow应用于经典机器学习

所有的机器学习问题都可以抽象成下面的等式：
$$
y = f(x)
$$
y是输出或者目标，x是输入或者特征。如果x是特征集合，那么可以称之为特征向量写成X。我们所说的模型，是指找到从特征到目标的映射函数f。一旦我们找到了f，我们就可以使用新的x来预测y值。

### 简单线性回归

#### 数据准备

我们使用sklearn包中的datasets模块里make_regression函数来生成数据集：

```python
from sklearn import datasets as skds
X, y = skds.make_regression(n_samples=200,
                           n_features=1,
                           n_informative=1,
                           n_targets=1,
                           noise=20.0)
>>> print(y.shape)
(200,)
```

因为生成的y是1维的Numpy数组，所以我们需要把它变成2维的数组

```python
if(y.ndim==1):
    y = y.reshape(len(y), 1)

>>> print(y.shape)
(200, 1)
```

通过画出生成数据集观察数据：

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 8))
plt.plot(X, y, 'b.')
plt.title('Original Dataset')
plt.show()
```

![Figure_1](media/Figure_1.png)

分割测试、训练数据集

```python
# 原书少了这个引用，而且过去sklearn中的cross_validation现在使用model_selection代替了
import sklearn.model_selection as skms
X_train, X_test, y_train, y_test = skms.train_test_split(X, y,
                                                        test_size=.2,
                                                        random_state=123)
```

#### 建立简单回归模型

使用TensorFlow建立简单回归模型，需要以下步骤：

1. 定义输入、参数和其他变量
2. 定义模型
3. 定义损失函数
4. 定义优化函数
5. 经过一定数量的迭代训练，也就是“代”

##### 定义输入、参数和其他变量

```python
num_outputs = y_train.shape[1]
num_inputs = X_train.shape[1]

x_tensor = tf.placeholder(dtype=tf.float32, shape=[None, num_inputs], name='x')
y_tensor = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs], name='y')

w = tf.Variable(tf.zeros([num_inputs, num_outputs]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros([num_outputs]), dtype=tf.float32, name='b')
```

##### 定义模型

```python
model = tf.matmul(x_tensor, w) + b
```

##### 定义损失函数

定义均方误差为损失函数（MSE）：
$$
\frac{1}{n}\sum(y_i-\hat{y_i})^2
$$

```python
loss = tf.reduce_mean(tf.square(model - y_tensor))
```

同时定义均方误差和r方（r-squared)函数用来评估训练模型。

```python
mse = tf.reduce_mean(tf.square(model - y_tensor))
y_mean = tf.reduce_mean(y_tensor)
total_error = tf.reduce_sum(tf.square(y_tensor - y_mean))
unexplained_error = tf.reduce_sum(tf.square(y_tensor-model))
rs = 1 - tf.div(unexplained_error, total_error)
```

##### 定义优化函数

```python
learning_rate = .01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

*注意写法*

##### 训练模型

现在有了模型，损失函数和优化函数，训练模型还需要定义以下全局变量：

* num_epochs：运行训练的迭代次数。每次迭代模型都会学习更好的参数，后续可以看图了解
* w_hat，b_hat：w、b参数的估计值
* loss_epochs, mse_epochs, rs_epochs：每次迭代的训练集中所有的误差值，同时还有测试集上的mse和r方
* mse_score和rs_score：最终模型的mse和r方的值

```python
import numpy as np
num_epochs = 1500
w_hat = 0
b_hat = 0
loss_epochs = np.empty(shape=[num_epochs], dtype=float)
mse_epochs = np.empty(shape=[num_epochs], dtype=float)
rs_epochs = np.empty(shape=[num_epochs], dtype=float)
loss_epoch = {}
mse_epoch = {}
rs_epoch = {}

mse_score = 0
rs_score = 0
```

初始化session和全局变量，运行训练循环num_epochs次

```python
with tf.Session() as tfs:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        # 每次循环都在训练集上运行优化
        tfs.run(optimizer, feed_dict={x_tensor: X_train, y_tensor: y_train})
        # 应用学习到的w、b计算误差，备将来画图
        loss_val = tfs.run(loss, feed_dict={x_tensor: X_train, y_tensor: y_train})
        loss_epochs[epoch] = loss_val
        # 依照测试数据集计算预测的MSE和R方
        mse_score = tfs.run(mse, feed_dict={x_tensor: X_test, y_tensor: y_test})
        mse_epochs[epoch] = mse_score
        #wh,bh = tfs.run([w, b])
        #print(wh, bh)
        #mh = tfs.run(model)
        rs_score = tfs.run(rs, feed_dict={x_tensor: X_test, y_tensor: y_test, w: wh, b: bh})
        #print(rs_score)
        rs_epochs[epoch] = rs_score
    # 最后循环一结束保存w,b
    #test1 = tfs.run(y_mean, feed_dict={y_tensor: y_test})
    #tErr = tfs.run(total_error, feed_dict={y_tensor: y_test, y_mean:test1})
    w_hat, b_hat = tfs.run([w, b])
    #test2 = tfs.run(model, feed_dict={x_tensor: X_test, w: w_hat, b: b_hat})
    #uErr = tfs.run(unexplained_error, feed_dict={y_tensor: y_test, model:test2})
    w_hat = w_hat.reshape(1)
    #mm=tfs.run(1-tf.div(uErr, tErr))
    #print(mm, uErr/tErr)

print('Model: Y = {0:.8f} X + {1:.8f}'.format(w_hat[0], b_hat[0]))
print('For test data: MSE = {0:.8f}, R2 = {1:.8f}'.format(mse_score, rs_score))
```

*两版tensorflow* 居然不同？

```python
plt.figure(figsize=(14, 8))
plit.title('Original Data and Trained Model')
x_plot = [np.min(X) - 1, np.max(X) + 1]
y_plot = w_hat*x_plot + b_hat
plt.axis([x_plot[0], x_plot[1], y_plot[0], y_plot[1]])
plt.plot(X, y, 'b.', label='Original Data')
plt.plot(x_plot, y_plot, 'r-', label='Trained Model')
plt.legend()
plt.savefig('figure1.png')
```

![figure1](media/figure1.png)

下面画出每代MSE训练和测试集的情况

```python
plt.figure(figsize=(14, 8))
plt.axis([0, num_epochs, 0, np.max(loss_epochs)])
plt.plot(loss_epochs, label='Loss on X_train')
plt.title('Loss in Iterations')
plt.xlabel('# Epoch')
plt.ylabel('MSE')

plt.axis([0, num_epochs, 0, np.max(mse_epochs)])
plt.plot(mse_epochs, label='MSE on X_test')
plt.xlabel('# Epoch')
plt.ylabel('MSE')
plt.legend()

plt.savefig('figure2.png')
```

![figure2](media/figure2.png)

我们看一下R方：

```python
plt.figure(figsize=(14, 8))
plt.axis([0, num_epochs, 0, np.max(rs_epochs)])
plt.plot(rs_epochs, label='R2 on X_test')
plt.xlabel('# Epoch')
plt.ylabel('R2')
plt.legend()
plt.savefig('figure3.png')
```

![figure3](media/figure3.png)

#### 使用训练好的模型预测

用线性模型预测，是在最小MSE的理解上而来的，我们看到，基本上是一条直线，与散点图拟合的不太完美。

### 多元回归















