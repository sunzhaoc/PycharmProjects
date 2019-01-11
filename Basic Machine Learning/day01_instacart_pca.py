import pandas as pd
orders = pd.read_csv("orders.csv")
aisles= pd.read_csv("aisles.csv")
order_products = pd.read_csv("order_products__prior.csv")
products = pd.read_csv("products.csv")
tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])
tab3
table = pd.crosstab(tab3["user_id"], tab3["aisle"])
table
data = table[:10000]
data
# 4、PCA降维
from sklearn.decomposition import PCA
# 1)实例化一个转换器类
transfer = PCA(n_components=0.95)
# 2)调用fit_transform
data_new = transfer.fit_transform(data)
data_new.shape
