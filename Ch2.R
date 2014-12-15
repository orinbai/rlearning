## Chapter 2: Start from Qplot ##
library(ggplot2)
set.seed(1410)
dsmall <- diamonds[sample(nrow(diamonds), 100),]
## First graphic for observing the correlation between carat and price ##
qplot(carat, price, data=diamonds)
## 在竖方向上有些条纹，所以可能是有指数关系 ##
qplot(log(carat), log(price), data=diamonds)

## params of qplot func can be combined of the variable ##
## We try to observe the relationship of carat and volumn ##
qplot(carat, x*y*z, data=diamonds)
## Observably, the density of diamond should be a constant. So carat and the volumn ##
## show a linear relation. ##

## Add color and shape to the graphic, something like groupby in mysql, and qplot 
## generate LEGEND automatically. ##
## 1. auto color ##
qplot(carat, price, data=dsmall, col=color)
## 2. auto shape ##
qplot(carat, price, data=dsmall, shape=cut)
## and of cause ##
qplot(carat, price, data=dsmall, col=color, shape=cut)
## But use I() specify attitude don't generate LEGEND ##
qplot(carat, price, data=dsmall, col='red')
qplot(carat, price, data=dsmall, col=I('red'))
## Use alpha to specify transparent ##
## alpha(1/n): total transparent - not transparent ##
##                 0           -        1          ##
## Denominator 'n' means after n times overlap, the graphic will be not transparent ##
## 1/10 mean after 10 times overlap, graphic will be not transparent ##
## 
qplot(carat, price, data=diamonds, alpha=I(1/10))
qplot(carat, price, data=diamonds, alpha=I(1/100))
qplot(carat, price, data=diamonds, alpha=I(1/200))

## We describe classifacioan variable with 'color' and 'shape', and use size for
## continuous variable. Massive Data maybe should use layer ##

## geom means geometry object, it describe which geometry object will be used to visualize
## the data. Some of them will correspond to several statistical transforms, like hist.
## geom = 'point' draws a scatter 
## geom = 'smooth' will fit and draw a smooth line, with it standard deviation ##
## geom = 'boxplot' draw a boxplot
## geom = 'path', geom = 'line' draw a line connected the data points.
## the direction of line must from left to right, path is free.
## geom = 'histogram' 直方图, geom='freqpoly' 概率多边形, geom = 'density' 密度曲线
## 只有x参数传递给qplot的话，默认画直方图.
## geom = 'bar', bar graphic

## add a smooth to show data trend ##
qplot(carat, price, data=dsmall, geom=c('point', 'smooth'))
qplot(carat, price, data=diamonds, geom=c('point', 'smooth'))
## when sample number below 1000, the loess will be default smooth func. we can try 'method='gam',
## formula=y ~ s(x)' to load package 'mgcv'. when sample number very large, try formula formula= y~
## s(x, bs='cs')
library('mgcv')
qplot(carat, price, data=dsmall, geom=c('point', 'smooth'), method='gam', formula=y~s(x))
qplot(carat, price, data=dsmall, geom=c('point', 'smooth'), method='gam', formula = y~s(x, bs='cs'))
## method = 'lm' always give you a line, unless specify formula=y~poly(x,2)
qplot(carat, price, data=dsmall, geom=c('point', 'smooth'), method='lm')
qplot(carat, price, data=dsmall, geom=c('point', 'smooth'), method='lm', formula=y~poly(x, 3))
## method = 'rlm' is like 'lm' but a more stable algorithms, but must loading package 'MASS' first.
library('MASS')
qplot(carat, price, data=dsmall, geom=c('point', 'smooth'), method='rlm')

## geom = 'jitter', 扰动图
qplot(color, price/carat, data=diamonds, geom='jitter', alpha=I(1/100))
## geom = 'boxplot', 箱线图
qplot(color, price/carat, data=diamonds, geom='boxplot', colour=I('blue'), fill=I('yellow'), size=1)

## geom = 'histogram', 直方图
qplot(carat, data=diamonds, geom='histogram', binwidth=0.1, xlim=c(0,3))
## geom = 'density', 密度图
qplot(carat, data=diamonds, geom='density')

## 分组 （这里使用color）##
qplot(carat, data=diamonds, geom='histogram', binwidth=0.5, xlim=c(0,3), fill=color, col=I('darkgray'))
qplot(carat, data=diamonds, geom='density', col=color)

## geom = 'bar', 会自动进行分组计算，不必事先汇总。如有必要，可以使用weight参数
qplot(color, data=diamonds, geom='bar')
## 按照weight汇总，计算不同color钻石的重量
qplot(color, data=diamonds, geom='bar', weight=carat) + scale_y_continuous("carat")

## 图形分窗格 ##
## 为了便于比较数据集中各个子类的关系，会想将子集数据分别绘制在不同的部分中。
## qplot的默认方法是划分窗格， 可以通过row var ~ col var的方式指定。当变量特别大的时候，
## 这么做不便于展示。如果指定一行或者一列，可以使用. 占位。比如，row var ~ .，是指定单列多行。
## 可以使用 facets = row var ~ col var来指定, facets就叫分面(facet panel)。
qplot(carat, data=diamonds, facets=color ~ ., geom='histogram', binwidth=0.1, xlim=c(0, 3))
## ..density.. 是新方法，告诉qplot使用频数而不是数量。
qplot(carat, ..density.., data=diamonds, facets=color~., geom='histogram', binwidth=0.1, xlim=c(0,3))

## 其他选项 ##
## xlim, ylim 是x,y坐标轴的范围，好像ggplot会画完整区间的坐标轴
## xlab, ylab 是x,y坐标轴的名字，可以使用表达式expression(formula)。
## main, 图标的名字，也可以使用表达式 expression(formula)。
## log, 表示那个轴取对数, log='x', 是x轴取对数，log='xy', 是俩轴都取对数。
qplot(carat, price/carat, data=dsmall, ylab=expression(frac(price, carat)))
qplot(carat, price, data=dsmall, log='xy')
