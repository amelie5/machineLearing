#对空值NaN的处理方法
import pandas as pd
df=pd.DataFrame([
           ['green','M',4,'class1'],
           ['red','L',7,'class2'],
           ['blue','XL',12,'class3']])
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)

#一列数据的映射方法
df=pd.DataFrame([
           ['green','M',4,'class1'],
           ['red','L',7,'class2'],
           ['blue','XL',12,'class3']])
df.columns=['color','size','price','classlabel']

size_mapping={
    'XL':3,
    'L':2,
    'M':1
}
df['size']=df['size'].map(size_mapping)
print(df)
inv_size_mapping={v:k for k,v in size_mapping.items()}
df['size_r']=df['size'].map(inv_size_mapping)
print(df)

#统一特征值取值范围（聚类、决策树、随机森林不需要）
#有两种方法：归一化和标准化
#归一化：缩放到【0，1】
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
x_train_norm=mms.fit_transform(x_train)
#标准化更实用：将特征值缩放到以0为中心，标准差为1的正太分布，这样学习权重参数更容易。
from sklearn.preprocessing import StandardScaler
stdsc=StandardScaler()
x_train_std=stdsc.fit_transform(x_train)

#特征选择:L1正则化与降低维度
#通过L1正则化进行特征选择，导致特征的稀疏性。L1的损失函数为统计权重参数绝对值的和，L2为平方和。
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')
lr=LogisticRegression(penalty='l1',C=0.1)
lr.fit(x_train_std,y_train)
print('Training accuracy:',lr.score(x_train,y_train))
print('Test accuracy:',lr.score(x_test,y_test))


#s使用SBS进行特征选择
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import SBS
knn=KNeighborsClassifier(n_neighbors=2)
sbs=SBS(knn,k_feature=1)
sbs.fit(x_train_std,y_train)

k_feat=[len(k) for k in sbs.subsets]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()