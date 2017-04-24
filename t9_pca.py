import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_wine=pd.read_csv()

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
x,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
sc=StandardScaler()
x_train_std=sc.fit_transform(x_train)
x_test_std=sc.transform(x_test)

#构建协方差 eig函数得到特征值和特征向量
import numpy as np
cov_mat=np.cov(x_train_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
print('\nEigenvalues \n{}'.format(eigen_vals))


#计算方差解释率
tot = sum(eigen_vals)
var_exp=[(i / tot) for i in sorted(eigen_vals,reverse=True)]
cum_var_exp=np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

