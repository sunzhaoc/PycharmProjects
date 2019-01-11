# 1、获取数据
# 2、合并表
# 3、找到user_id和aisle_id之间的关系
# 4、PCA降维

import pandas as pd

# 1、获取数据
orders = pd.read_csv("orders.csv")
aisles= pd.read_csv("aisles.csv")
order_products = pd.read_csv("order_products__prior.csv")
products = pd.read_csv("products.csv")

# 2、合并表
# order_products__prior.csv：订单与商品信息
# 字段：order_id	product_id	add_to_cart_order	reordered

# products.csv：商品信息
# 字段：product_id	product_name	aisle_id	department_id

# orders.csv：用户的订单信息
# 字段：order_id	user_id	eval_set	order_number	order_dow	order_hour_of_day	days_since_prior_order

# aisles.csv：商品所属具体物品类别
# 字段：aisle_id	aisle

# 合并aisles和products
tab1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])
tab2 = pd.merge(tab1, order_products, on=["product_id", "product_id"])
tab3 = pd.merge(tab2, orders, on=["order_id", "order_id"])
tab3.head()

# 3、找到user_id和aisle_id之间的关系
table = pd.crosstab(tab3["user_id"], tab3["aisle"])

data = table[:10000]

# 4、PCA降维
from sklearn.decomposition import PCA

# 1)实例化一个转换器类
transfer = PCA(n_components=0.95)
# 2)调用fit_transform
data_new = transfer.fit_transform(data)
data_new.shape
