library(ggplot2)
## 如果分组变量不是factor, legend会以连续变量作为color
qplot(displ, hwy, data=mpg, col=cyl)
## 如果是factor
qplot(displ, hwy, data=mpg, col=factor(cyl))

## Aesthetics（图形属性）就是位置(纵横坐标), 大小、颜色、形状等, 这些属性可以映射成一个
## 一个变量或者常数。
## 而像点(point), 线(line), 条(bar)都形状，被称作geom。一般geom决定了图形的名称。

## 实际数据中的单位(吨、小时等), 对于电脑而言无意义，需要将其转化为电脑懂的物理单位(像素、颜色）。
## 这个过程被称作尺度转换(scaling)。比如颜色是'FFFFFF'，这样的16进制数字。
## 当我们转换坐标时, 实际上是将数字从数字单位建立线性映射到[0,1]区间, 我们并没有使用像素。
## 最后根据点都位置确定它在图中都位置，这就是坐标体系(coord)都事情了。

## 一个完整的图形需要：数据，由点来表示；标度和坐标系，用来生成坐标轴和图例；图表注释，背景标题。
## 一个例图：比之前提到都组成多了三个新组建，分面、多图层、统计量。
qplot(displ, hwy, data=mpg, facets=.~year)+geom_smooth(method='loess')
## 每个图层上的分面(facet panel), 都含有一个小的数据子集，所以我们可以将这种方法开做三维矩阵。
## 分面是两维网络， 图层是在第三个维度的方向上叠加。

