 
import pandas 
 

data = pandas.read_csv(
    'BCHAIN-MKPRU.csv' ,
    index_col='Date'
)
#---------
#这里需要把文件放到和程序所在的文件夹
#---------
# 绘图过程中
 
import  matplotlib.pyplot as plt
 
# 用来正常显示中文标签
 
plt.rcParams['font.sans-serif']=['SimHei']
 
# 用来正常显示负号
 
plt.rcParams['axes.unicode_minus'] = False 
 
# 查看趋势图
data.plot() #有增长趋势，不平稳
 
 
# 附加：查看自相关系数合片自相关系数（查分之后），可以用于平稳性的检测，也可用于定阶系数预估
 
#自相关图（）
 
from statsmodels.graphics.tsaplots import plot_acf
 
plot_acf(data).show() #自相关图既不是拖尾也不是截尾。以上的图的自相关是一个三角对称的形式，这种趋势是单调趋势的典型图形，说明这个序列不是平稳序列
 
 
 
 
 
# 1 平稳性检测
 
from statsmodels.tsa.stattools import adfuller as ADF
 
 
def tagADF(t):
    result = pandas.DataFrame(index=[
            "Test Statistic Value", "p-value", "Lags Used", 
            "Number of Observations Used", 
            "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"
        ], columns=['Value']
    );
    result['Value']['Test Statistic Value'] = t[0]
    result['Value']['p-value'] = t[1]
    result['Value']['Lags Used'] = t[2]
    result['Value']['Number of Observations Used'] = t[3]
    result['Value']['Critical Value(1%)'] = t[4]['1%']
    result['Value']['Critical Value(5%)'] = t[4]['5%']
    result['Value']['Critical Value(10%)'] = t[4]['10%']
    return result;
 
 
print('原始序列的ADF检验结果为:',tagADF(ADF(data[u'Value'])))  # 添加标签后展现
 
# 平稳判断：得到统计量大于三个置信度(1%,5%,10%)临界统计值，p值显著大于0.05，该序列为非平稳序列。
# 备注：得到的统计量显著小于3个置信度（1%，5%，10%）的临界统计值时，为平稳 此时p值接近于0 此处不为0，尝试增加数据量，原数据太少
 
# 2 进行数据差分，一般一阶差分就可以
 
D_data = data.diff(1).dropna()
D_data.columns = [u'Value差分']
 
#差分图趋势查看
 
D_data.plot() 
plt.show()
 
# 附加：查看自相关系数合片自相关系数（查分之后），可以用于平稳性的检测，也可用于定阶系数预估
 
#自相关图
 
plot_acf(D_data).show()
 
plt.show()
 
#偏自相关图
 
from statsmodels.graphics.tsaplots import plot_pacf
 
plot_pacf(D_data).show()
 
# 3 平稳性检测
 
print(u'差分序列的ADF检验结果为：', tagADF(ADF(D_data[u'Value差分']))) 
 
# 解释：Test Statistic Value值小于两个水平值，p值显著小于0.05，一阶差分后序列为平稳序列。
 
# 4 白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
 
#返回统计量和p值
 
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 分别为stat值（统计量）和P值
 
# P值小于0.05，所以一阶差分后的序列为平稳非白噪声序列。
 

# 5 p，q定阶
 
from statsmodels.tsa.arima_model import ARIMA
'''
pmax = int(10) 

qmax = int(10) 
 
#bic矩阵
 
bic_matrix = [] 
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
#存在部分报错，所以用try来跳过报错。
    try: 
      tmp.append(ARIMA(data, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)
 
#从中可以找出最小值
 
bic_matrix = pandas.DataFrame(bic_matrix) 
 
#先用stack展平，然后用idxmin找出最小值位置。
 
p,q = bic_matrix.stack().idxmin() 
 
 
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))
# 取BIC信息量达到最小的模型阶数
'''

#----------
#第五部分由于运行时间的原因，暂且跳过，下面我直接取运行的结果，避免每次运行都要花很长时间(那个6 1 4 不要随意改动 会出错)
#----------
# 6 建立模型和预测
model = ARIMA(data, (6,1,4)).fit() 
 
#给出一份模型报告（在输出处 不是图片）
 
model.summary2() 
 
#预测，返回预测结果、标准误差、置信区间。这个的作用是预测后面几天的 如果是我们这种全部预测的就不需要用这个了
#pre = model.forecast(1)[0]
model.plot_predict(dynamic=False)
plt.show()
 
 
 