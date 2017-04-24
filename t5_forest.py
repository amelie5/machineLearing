from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)
forest.fit()