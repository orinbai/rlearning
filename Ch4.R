#Chap 4 #
#--- Oxboys data in library: nlme ---##
library(nlme)
## 图层由5个层面组成：
## 数据，必须是一个数据框
## 一组图形属性映射，用来设定数据集中的变量如何映射到该图层的图形属性。
## 几何对象， 指定图层中用哪些几何对象绘图
## 统计变换， 对数据进行有用都统计处理（比如光滑）。
## 位置调整， 通过调整元素位置避免重叠。
## 实际上，表现是mapping, data, geom, statistics transfrom, position adjusting

## 创建绘图对象， ggplot有俩参数，一个是数据，一个是图形属性。
p <- ggplot(diamonds, aes(carat, price, col=color))
## 再增加一个几何对象
p <- p + layer(geom = 'point')
## 其实使用了隐含的默认值：统计变换和位置调整。
## layer(geom, geom_params, stat, stat_params, data, mapping, position) 也是5组成的体现
## 比较完整的图层写法： 
p <- ggplot(diamonds, aes(carat))
p <- p + layer(
  geom = 'bar',
  geom_params = list(fill='steelblue'),
  stat = 'bin',
  stat_params = list(binwidth=2)
)

## layer写法太复杂， 所以尝试快捷写法， 一般由geom_xxx, stat_xxx组成
## geom_xxx(mapping, data, ..., stat, position)
## stat_xxx(mapping, data, ..., stat, position)
p <- ggplot(diamonds, aes(carat))
p <- p + geom_histogram(binwidth=0.2, fill='steelblue')
summary(p)
## Summary(p), 可以清晰的看到5层结构

## Show: ##
library(scales)
bestfit <- geom_smooth(method='lm', se=F, col=alpha('steelblue', 0.5), size=2)
qplot(sleep_rem, sleep_total, data=msleep) + bestfit
qplot(awake, brainwt, data=msleep, log='y') + bestfit
qplot(bodywt, brainwt, data=msleep, log='xy') + bestfit

## 比较有趣的情况，aes中colour参数一般是用作映射，而不是赋值。图层中的则相反，
## 以下例子可以看得比较明显。
p <- ggplot(mtcars)
## 1. aes中定义colour，第一图形颜色并非darkblue，这是因为这里是映射而不是赋值，所以
## 相当于ggplot自己去按照颜色选取原则在色盘上选择了一个颜色，darkblue不过是个字符串，什么
## 值都可以，其实相当于 col = c('darkblue')。
p + geom_line(aes(mpg, wt, col='darkblue'))
p + geom_line(aes(mpg, wt, col=c('darkblue'))
p + geom_line(aes(mpg, wt, col='aaaa'))
## 而在图层参数中赋值就完全不同了，就是颜色本身。
## 以上都不同主要因为：图形属性表映射，x=X, y=Y，意思是把数据X、Y分别映射到坐标轴X、Y上，
## colour也是如此，这时colour = 'darkblue'其实不是赋值，而是colour映射到字串'darkblue'上。
## 而图层属性表示赋值，所以就对了。
p + geom_line(aes(mpg, wt), col='darkblue')

## 多图层可以重新组织、映射数据
## Oxboys 数据以阶段作为分类变量（因为Occassion是离散变量，会自动变成分类变量），画箱线图
p <- ggplot(Oxboys, aes(Occasion, height)) + geom_boxplot()
## 在同一个坐标体系中添加每个男孩在不同阶段都变化line，也就是以Subject为group 画 geom_line
p + geom_line(aes(group=Subject), col='#3366FF')

## 每个几何对象有一种默认的统计变换，并且每一个统计变换都有一个默认的几何对象，这话反复出现多次
## 所以统计变换也应该是一个重要概念。
## 统计变化，stat，通常是以某种方式对数据进行信息汇总。为了阐明在图中的意义，统计变换必须是一个
## 位置尺度不变的量，即f(x+a) = f(x)+a 且 f(b*x) = b*f(x)。这样才能保证，在改变图形的标度时， 数据
## 变换保持不变。
## 统计变换实际上是数据集都重新生成，所以理论上统计变换可以产生新的变量。比如histogram图，在统计变
## 换时会产生 count(频数), density(密度), x(中心) 都变量。这些变量被称作生成变量，可以被直接调用。
ggplot(diamonds, aes(carat)) + geom_histogram(aes(y=..density..), binwidth=0.1)
## 等价于
qplot(carat, ..density.., data=diamonds, geom = 'histogram', binwidth=0.1)

## 属性组合演示 ##
d <- ggplot(diamonds, aes(carat)) + xlim(0,3)
d + stat_bin(aes(ymax=..count..), binwidth=0.1, geom='area')
d + stat_bin(aes(size = ..density.., ymax=..count..), binwidth=0.1, geom='point', position='identity')
d + stat_bin(aes(y=1, fill=..count..), binwidth=0.1, geom='tile', position='identity')
## 最后这个是错的，主要是因为 stat_bin会对数据做转换。而生成Y，这时，如果手动指定Y，就会出错了，
## 如果一定要指定，只能是bin统计变换的结果，比如 ..x.., ..count.., ..density..

## 绘制多数据源 ##
require(nlme, quiet=T, warn.conflict=F)
model <- lme(height~age, data=Oxboys, random=~1+age|Subject)
oplot <- ggplot(Oxboys, aes(age, height, group=Subject))+geom_line()
age_grid <- seq(-1, 1, length=10)
## 使用相同的变量名，可以在绘图时继续使用原始的分类属性 ##
subjects <- unique(Oxboys$Subject)
preds <- expand.grid(age = age_grid, Subject=subjects)
preds$height <- predict(model, preds)
oplot + geom_line(data=preds, col='#3366FF', size=0.4)

## 这张残差图显示残差并不随机，并且似乎是2次函数，所以增加一个age的2次项 ##
model2 <- update(model, height ~ age + I(age^2))
Oxboys$fitted2 <- predict(model2)
Oxboys$resid2 <- with(Oxboys, fitted2 - height)
## 现在残差几乎随机了 ##


