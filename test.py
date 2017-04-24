import pandas as pd
import numpy as np

df=pd.DataFrame([
           ['green','M',4,'class1'],
           ['red','L',7,'class2'],
           ['blue','XL',12,'class3']])
df.columns=['color','size','price','classlabel']

size_mapping={
    4:'haha',
    7:'no'


}
df['price_new']=df['price'].map(size_mapping)
print(df)


