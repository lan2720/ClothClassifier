#coding:utf-8
"""a demo of matplotlib"""
import matplotlib as  mpl
from matplotlib.font_manager import FontProperties
chinese_font = FontProperties(fname='yahei.ttf')
from matplotlib import pyplot as plt
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
#创建一副线图,x轴是年份,y轴是gdp
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
#添加一个标题
plt.title(u'名义GDP', fontproperties=chinese_font)
#给y轴加标记
plt.ylabel(u'十亿美元', fontproperties=chinese_font)
plt.savefig('hehe.png')

