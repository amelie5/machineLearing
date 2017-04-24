from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
#metric为选择距离都的算法，此处的minkowski是欧式和曼哈顿距离的一般化，若p=2则退化为欧氏距离，p=1则为曼哈顿距离
knn.fit()