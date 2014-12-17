## Toolkit ##
## 有不少例子可以看看 ##
## import library ##
library('ggplot2')
library('scales')
library('hexbin')
dsmall <- diamonds[sample(nrow(diamonds), 300),]
df <- data.frame(
  x <- c(3, 1, 5),
  y <- c(2, 4, 5),
  label <- c('a', 'b', 'c'))

p <- ggplot(df, aes(x, y))+xlab(NULL)+ylab(NULL)
p + geom_point() + labs(title='geom point')
p + geom_line() + labs(title='geom line')
p + geom_bar(stat='identity') + labs(title='geom_bar(stat=\'identity\')') 
p + geom_area() + labs(title='geom_area')
p + geom_path() + labs(title='geom_path')
p + geom_text(aes(label=label)) + labs(title='geom_text')
p + geom_tile() + labs(title='geom_tile')
p + geom_polygon() + labs(title='geom_polygon')

## Distribution Visualization ##
## histogram and freqpoly use stat='bin' as default, so there are two variables automatic computed.
## 'count' and 'density', density almost equal count/total number of count ##
depth_dist <- ggplot(diamonds, aes(depth)) + xlim(58, 68)
depth_dist + geom_histogram(aes(y=..density..), binwidth=0.1) + facet_grid(cut ~ .)
depth_dist + geom_histogram(aes(fill=cut), binwidth=0.1, position='fill')
depth_dist + geom_freqpoly(aes(y=..density.., col=cut), binwidth=0.1)

## geom_boxplot = stat_boxplot + geom_boxplot, 连续型变量要先进行封箱(bin)操作再进行boxplot
library(plyr)
## 离散型变量的boxplot ##
qplot(cut, depth, data=diamonds, geom='boxplot')

## 对连续型变量，使用 round_any 进行封箱(bin)处理后，并指定group，绘制boxplot。round_any是一种精确度限制
## 的模式，round_any(carat, 0.1, floor)， 是将carat的精度限制在0.1，这样天然形成bin
qplot(carat, depth, data=diamonds, geom='boxplot', group=round_any(carat, 0.1, floor), xlim=c(0,3))

## geom_jitter = position_jitter + geom_point, 在离散型变量分布上绘制随机噪声点，以解决遮盖问题 ##
qplot(class, cty, data=mpg, geom='jitter')
qplot(class, drv, data=mpg, geom='jitter')

## geom_density=stat_density + geom_area，基于核平滑方法进行平滑后得到都多边形， 请在一直密度函数连续、
## 平滑、无界的情况下使用。实际上，是直方图的平滑版本，虽然有良好的理论性质，但很难由图回溯到数据。
qplot(depth, data=diamonds, geom='density', xlim=c(54, 70))
qplot(depth, data=diamonds, geom='density', xlim=c(54, 70), fill=cut, alpha=I(0.2))

## overplotting 问题处理 ##
df <- data.frame(x=rnorm(2000), y=rnorm(2000))
norm <- ggplot(df, aes(x, y))
norm + geom_point()
norm + geom_point(shape=1)
norm + geom_point(shape='.')
## 使用alpha来解决一部分遮盖问题，但由于alpha的比值是指多少次重叠为不透明，而且R最多支持1/256
## 所以很容易不够。
norm + geom_point(col='black', alpha=1/10)
norm + geom_point(col='black', alpha=1/3)

## jitter的技巧：如果变量稍微有点离散，那么jitter可能会在边界流出空白，更便于监测。
## position_jitter 从图上看，应该是将部分点平移了。具体待查。
td <- ggplot(diamonds, aes(table, depth)) + xlim(50, 70) + ylim(50, 70)
td + geom_point()
td + geom_jitter()
jit <- position_jitter(width = 0.9)
td + geom_jitter(position=jit)
td + geom_jitter(position=jit, col='black', alpha=0.1)
td + geom_jitter(position=jit, col='black', alpha=0.05)
td + geom_jitter(position=jit, col='black', alpha=1/200)

## 上面的做法感觉是二维核密度估计， 所以我们可以尝试先统计分箱中点的数量， 然后用图形表达出来
## 选择六边形。
d <- ggplot(diamonds, aes(carat, price)) + xlim(1, 3) + theme(legend.position="none")
d + stat_bin2d()
d + stat_bin2d(bins=10)
d + stat_bin2d(binwidth=c(0.02, 200))
d + stat_binhex()
d + stat_binhex(bins=10)
d + stat_binhex(binwidth=c(0.02, 200))
d + geom_point() + geom_density2d()
d + stat_density2d(geom='point', aes(size=..density..), contour=F,) + scale_size_area()
d + stat_density2d(geom='tile', aes(fill = ..density..), contour=F)
last_plot() + scale_fill_gradient(limits=c(1e-5, 8e-4))

## Map Visual ##
library(maps)
data(us.cities)
big_cities <- subset(us.cities, pop>500000)
qplot(long, lat, data=big_cities) + borders("state", size=0.1)+geom_text(label=big_cities$name)

tx_cities <- subset(us.cities, country.etc=='TX')
ggplot(tx_cities, aes(long,lat)) + borders("county", "texas", col='gray70') + geom_point(col='black', alpha=0.5)

## choropleth map ##
## 使用特定的列将地图数据和实际数据结合（使用merge函数）##

states <- map_data('state')
arrests <- USArrests
names(arrests) <- tolower(names(arrests))
arrests$region <- tolower(rownames(USArrests))
choro <- merge(states, arrests, by='region')
choro <- choro[order(choro$order),]
qplot(long, lat, data=choro, group=group, fill=assault, geom='polygon')
qplot(long, lat, data=choro, group=group, fill=assault/murder, geom='polygon')

## 需要用到 library(plyr) ##
## ddply(.data, .variables, .func=NULL) 其中： ##
## .data是默认数据集， .variables是用来做处理的分组变量. .func是对应的函数 ##
## 下面的例子里，ddply(ia, .(subregion), colwise(mid_range, .(lat, long)))中： ##
## ia数据框的subregion变量为分组变量（group），使用colwise对所有subregion相同的 ##
## 数据样本分别求其经纬度的平均值， 该值就是subregion组的平均值。 ##
## 对应到数据含义中，subregion是郡名， 所以centres其实就是某个郡的中心。  ##
## ---------------------------------------------------------------------  ##
## 从ggplot绘图过程中，可以看出同时使用两个数据集做图，那么需要aes中某些变量名一致 ##
## 测试时添加了几个变量:"lat1, long1, ooo", 根据变量名去掉这些列时，使用了match函数 ##
## ia[, -match(delete.names, colnames(ia))] ##

library(plyr)
ia <- map_data('county', 'iowa')
mid_range <- function(x) {
  mean(range(x, na.rm=T))
}
centres <- ddply(ia, .(subregion), colwise(mid_range, .(lat, long)))
ggplot(ia, aes(lat, long)) + geom_polygon(aes(group=group), fill=NA, col='gray70') + geom_text(aes(lat1, long1, label=subregion), data=centres, size=3, angle=45)

## 揭示不确定性 ##
## rbinom(nrow(diamonds), 1, 0.2) 意思是 二项分布中1的概率是0.2 ##
## d 都使用方式其实等同于 20% 抽样 ##

d <- subset(diamonds, carat<2.5 & rbinom(nrow(diamonds), 1, 0.2) == 1)
d$lcarat <- log(d$carat)
d$lprice <- log(d$price)
model <- lm(lprice ~ lcarat, data=d)
## 去线性回归残差值为新的Y，其实是指预测值无法被线性关系解释的部分（剔除线性关系后的变量自变量 ##
## 关系 ##
d$lprice2 <- resid(model)
mod <- lm(lprice2 ~ lcarat*color, data=d)
library(effects)
effectdf <- function(...) {
  suppressWarnings(as.data.frame(effect(...)))
}
## color 在mod中的fit，se， 取值区间等参数,表达mod的边际效应
color <- effectdf("color", mod)
## lcarat:color交互效应的边际效应
both1 <- effectdf("lcarat:color", mod)

carat <- effectdf('lcarat', mod, default.levels=50)
both2 <- effectdf('lcarat:color', mod, default.levels=3)

## 通过数据变换移除明显效应，lcarat 对lprice、lprice2的边际效应的情况。x，y取对数，去除了一部分非线性趋势
qplot(lcarat, lprice, data=d, col=color)
## 去除主要线性效应(lm建立的model)
qplot(lcarat, lprice2, data=d, col=color)

fplot <- ggplot(mapping=aes(y=fit, ymin=lower, ymax=upper)) + ylim(range(both2$lower, both2$upper))
## color的边际效应 ##
fplot %+% color + aes(x=color) + geom_point() + geom_errorbar()
## color在lcarat水平下的条件效应 ##
fplot %+% both2 + aes(x=color, col=lcarat, group=interaction(color, lcarat)) + geom_errorbar() +
  geom_line(aes(group= lcarat)) + scale_color_gradient()

## carat的不确定性 ##
fplot %+% carat + aes(x=lcarat) + geom_smooth(stat='identity')

## 针对color的不同水平，carat的条件效应 ##
ends <- subset(both1, lcarat=max(lcarat))
fplot %+% both1 + aes(x=lcarat, col=color) + geom_smooth(stat='identity') + scale_colour_hue() +
  theme(legend.position="none") + geom_text(aes(label=color, x=lcarat+0.02), ends)

## stat_summary ##

midm <- function(x) {
  mean(x, trim = 0.5)
}
m2 <- ggplot(movies, aes(factor(round(rating)), log10(votes)))
## 说明stat_summary 的fun.y参数可以接受简单的数学函数，回传一个值给Y ##
m2 + 
  stat_summary(aes(colour='trimmed'), fun.y=midm, geom = 'point') +
  stat_summary(aes(colour = 'raw'), fun.y=mean, geom = 'point') +
  scale_color_hue("Mean")

## 演示stat_summary 的fun.data可以使用更加复杂的函数，该函数返回一个各元素有名称的向量 ##
## 这种返回也许可以作为参数传递 ##
## stat_summary 可以搭配 Hmisc 包中的摘要函数使用 ##
m <- ggplot(movies, aes(year, rating))
iqr <- function(x, ...) {
  qs <- quantile(as.numeric(x), c(0.25, 0.75), na.rm=T)
  names(qs) <- c('ymin', 'ymax')
  qs
}
m + geom_ribbon()
m + stat_summary(fun.data='iqr', geom='ribbon')

## 添加图形注解 ##
## 其实，这些注解也是数据，可以采用逐个添加也可以采用批量添加的模式 ##
(unemp <- qplot(date, unemploy, data=economics, geom = 'line', xlab='', ylab='No. unemployed (1000s)'))
presidential <- presidential[-(1:3),]
yrng <- range(economics$unemploy)
xrng <- range(economics$date)
## 这里绘制平行于y轴的直线时，需要将日期转化成数字 ##
unemp + geom_vline(aes(xintercept=as.numeric(start)), data=presidential)
unemp + geom_rect(aes(NULL, NULL, xmin=start, xmax=end, fill=party), ymin=yrng[1], ymax=yrng[2], data=presidential, alpha=0.2) + scale_fill_manual(values=c('blue', 'red'))

last_plot() + geom_text(aes(start, yrng[1], label=name),data=presidential, size=3, hjust=0, vjust=0)
